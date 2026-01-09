# app3.py
# Sen2Fire — Transparent Pixel-wise Inference (2-model selector)
# No sidebar, centered input, model choice activates selected weights only.
# Pipeline: logits -> sigmoid -> threshold (validation-derived) -> visualization/product
# Ground-truth removed (per request).

import io
import time
import inspect
from pathlib import Path

import numpy as np
import torch
import streamlit as st

# pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp


# =========================
# MODEL REGISTRY (EDIT IF NEEDED)
# =========================
APP_DIR = Path(__file__).resolve().parent

MODEL_REGISTRY = {
    "U-Net (JP retrained)": {
        "weights": APP_DIR / "thesis_unet" / "unet_best_retrained2.pth",
        "t_best": 0.15,
        "rgb_idxs": (0, 1, 2),
        # Normalization placeholders (replace later with JP values)
        "mean_13": [0.0] * 13,
        "std_13": [1.0] * 13,
        # Architecture config
        "arch": "unet",
        "encoder_name": "resnet18",
        "encoder_weights": "imagenet",
        "in_channels": 13,
        "classes": 1,
    },
    "U-Net Eff (EfficientNet-B4 pretrained)": {
        "weights": APP_DIR / "unet_eff" / "efficientb4_pretrained.pth",
        "t_best": 0.15,   # keep fixed for now; adjust later if you have a different val-derived threshold
        "rgb_idxs": (0, 1, 2),
        "mean_13": [0.0] * 13,
        "std_13": [1.0] * 13,
        "arch": "unet",
        "encoder_name": "efficientnet-b4",
        "encoder_weights": "imagenet",
        "in_channels": 13,
        "classes": 1,
    },
        "DeepLabV3+ (EfficientNet-B4)": {
        "weights": APP_DIR / "thesis_deeplab" / "deeplab-b4.pth",
        "t_best": 0.15,          # sementara sama, bisa beda nanti
        "rgb_idxs": (0, 1, 2),
        "mean_13": [0.0] * 13,   # placeholder, sama seperti model lain
        "std_13": [1.0] * 13,
        "arch": "deeplab",
        "encoder_name": "efficientnet-b4",
        "encoder_weights": "imagenet",
        "in_channels": 13,
        "classes": 1,
    },

}


# =========================
# CORE HELPERS
# =========================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_batch(x: torch.Tensor, mean_13, std_13) -> torch.Tensor:
    mean = torch.tensor(mean_13, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std = torch.tensor(std_13, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - mean) / (std + 1e-6)


def build_model(cfg: dict):
    arch = cfg["arch"].lower()

    if arch == "unet":
        return smp.Unet(
            encoder_name=cfg["encoder_name"],
            encoder_weights=cfg["encoder_weights"],
            in_channels=cfg["in_channels"],
            classes=cfg["classes"],
            activation=None,  # logits
        )

    if arch == "deeplab":
        return smp.DeepLabV3Plus(
            encoder_name=cfg["encoder_name"],
            encoder_weights=cfg["encoder_weights"],
            in_channels=cfg["in_channels"],
            classes=cfg["classes"],
            activation=None,  # logits
        )

    raise ValueError(f"Unsupported arch: {cfg['arch']}")


@st.cache_resource
def load_model_cached(weights_path_str: str, cfg_fingerprint: str):
    """
    Cache key includes:
      - weights_path_str
      - cfg_fingerprint (to avoid collisions if arch/encoder changes)
    """
    device = get_device()
    cfg = MODEL_REGISTRY[cfg_fingerprint]
    model = build_model(cfg).to(device)

    load_kwargs = {"map_location": device}
    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        load_kwargs["weights_only"] = False  # trusted checkpoint

    ckpt = torch.load(weights_path_str, **load_kwargs)

    state = None
    if isinstance(ckpt, dict):
        for k in ["model_state", "state_dict", "model_state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                state = ckpt[k]
                break
        if state is None:
            state = ckpt
    else:
        state = ckpt

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_npz_from_bytes(npz_bytes: bytes):
    """
    Required keys:
      image: (12,H,W)
      aerosol: (H,W)
    Optional key:
      label: ignored in this version
    Returns:
      x13: (13,H,W) float32
    """
    bio = io.BytesIO(npz_bytes)
    with np.load(bio) as data:
        keys = list(data.files)
        if "image" not in keys or "aerosol" not in keys:
            raise KeyError(f"NPZ missing required keys. Found keys: {keys}")

        img = data["image"].astype(np.float32)
        aer = data["aerosol"].astype(np.float32)

    if img.ndim != 3:
        raise ValueError(f"`image` must be 3D (C,H,W). Got shape {img.shape}")
    if aer.ndim != 2:
        raise ValueError(f"`aerosol` must be 2D (H,W). Got shape {aer.shape}")

    aer = aer[None, ...]
    x13 = np.concatenate([img, aer], axis=0)
    return x13


def to_rgb_for_display(x13: np.ndarray, rgb_idxs=(0, 1, 2)):
    rgb = np.stack([x13[rgb_idxs[0]], x13[rgb_idxs[1]], x13[rgb_idxs[2]]], axis=-1)
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        ch = rgb[..., c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        if hi - lo < 1e-6:
            out[..., c] = 0.0
        else:
            out[..., c] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out


def overlay_mask(rgb01: np.ndarray, mask01: np.ndarray, alpha=0.35):
    rgb = rgb01.copy()
    red = np.zeros_like(rgb)
    red[..., 0] = 1.0
    m = mask01[..., None].astype(np.float32)
    rgb = (1 - alpha * m) * rgb + (alpha * m) * red
    return np.clip(rgb, 0, 1)


def prob_to_heatmap_gray(prob: np.ndarray):
    p = np.clip(prob.astype(np.float32), 0, 1)
    return np.stack([p, p, p], axis=-1)


# =========================
# PAGE CONFIG + STYLING
# =========================
st.set_page_config(page_title="Sen2Fire (Pixel-wise) — 2 Models", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"]  { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
.block-container { padding-top: 1.5rem; }

/* Center container like a product landing page */
.center-wrap { max-width: 980px; margin: 0 auto; }

/* Upload card */
.upload-card {
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.03);
    border-radius: 18px;
    padding: 18px 18px 14px 18px;
}

.center-muted {
    text-align: center;
    color: rgba(255,255,255,0.70);
    font-size: 0.95rem;
    line-height: 1.4;
    margin-top: 10px;
}

.kv-center {
    margin-top: 12px;
    display: grid;
    grid-template-columns: 260px 1fr;
    gap: 6px 12px;
    font-size: 0.95rem;
}
.kv-center .k { color: rgba(255,255,255,0.65); }
.kv-center .v {
    color: rgba(255,255,255,0.95);
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
}

.card {
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    padding: 14px 14px 10px 14px;
}

hr { border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 18px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# =========================
# HEADER
# =========================
st.title("Sen2Fire — Transparent Pixel-wise Inference (2 Models)")
st.markdown("Segmentation (logits) → sigmoid → data-driven threshold (validation-derived) → visualization & product.")

# =========================
# CENTERED INPUT
# =========================
st.markdown('<div class="center-wrap">', unsafe_allow_html=True)
st.markdown('<div class="upload-card">', unsafe_allow_html=True)
st.markdown("## Input")

# model selection (centered)
mcol1, mcol2, mcol3 = st.columns([1, 2.2, 1])
with mcol2:
    model_name = st.selectbox(
        "Choose model",
        list(MODEL_REGISTRY.keys()),
        index=0,
        label_visibility="collapsed",
    )

cfg = MODEL_REGISTRY[model_name]
weights_path: Path = cfg["weights"]
t_best: float = float(cfg["t_best"])
rgb_idxs = tuple(cfg["rgb_idxs"])
mean_13 = list(cfg["mean_13"])
std_13 = list(cfg["std_13"])

# file upload (centered)
up_col_left, up_col_mid, up_col_right = st.columns([1, 2.2, 1])
with up_col_mid:
    uploaded = st.file_uploader("Upload 1 test patch (.npz)", type=["npz"], label_visibility="collapsed")

st.markdown(
    """
<div class="center-muted">
Accepted: <b>.npz</b> • Recommended patch size: <b>512×512</b><br/>
Required keys: <code>image</code>, <code>aerosol</code>
</div>
""",
    unsafe_allow_html=True,
)

# model info
st.markdown(
    f"""
<div class="kv-center">
  <div class="k">Selected model</div><div class="v">{model_name}</div>
  <div class="k">Model weights</div><div class="v">{weights_path.name}</div>
  <div class="k">Device</div><div class="v">{str(get_device())}</div>
  <div class="k">Threshold (val-derived)</div><div class="v">{t_best}</div>
  <div class="k">RGB composite (display)</div><div class="v">{rgb_idxs}</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("</div>", unsafe_allow_html=True)

with st.expander("Input specification (what this app expects)"):
    st.markdown(
        """
- File format: `.npz` (NumPy zipped arrays)
- Required arrays:
  - `image`: `(12, H, W)` (Sentinel-2 bands in dataset-defined order)
  - `aerosol`: `(H, W)` (single-channel aerosol layer)
- Expected patch size: `H=W=512` (app will still run on other sizes if the model supports it)
- Model input tensor: `x = concat(image, aerosol[None,:,:])` ⇒ `(13, H, W)`
- Model output: logits `(1, H, W)` → sigmoid probability `(1, H, W)` → threshold → binary mask `(H, W)`
        """.strip()
    )

st.markdown("</div>", unsafe_allow_html=True)

# Warnings about placeholder normalization
if mean_13 == [0.0] * 13 and std_13 == [1.0] * 13:
    st.warning(
        f"Normalization for **{model_name}** is currently PLACEBO (mean=0, std=1). "
        "OK for UI/pipeline wiring. Replace later with JP’s MEAN_13/STD_13 for accurate calibration."
    )

# Validate weights exist
if not weights_path.exists():
    st.error(f"Weights not found for selected model: {weights_path}")
    st.stop()

if len(mean_13) != 13 or len(std_13) != 13:
    st.error("MEAN_13 and STD_13 must each have length 13 for the selected model.")
    st.stop()

if uploaded is None:
    st.info("Choose a model, then upload a .npz file to start.")
    st.stop()

# =========================
# LOAD NPZ + INFER
# =========================
try:
    x13 = load_npz_from_bytes(uploaded.getvalue())
except Exception as e:
    st.error(f"Failed to read NPZ: {e}")
    st.stop()

H, W = x13.shape[1], x13.shape[2]
size_ok = (H == 512 and W == 512)

device = get_device()
try:
    # cache keyed by (weights path, model_name)
    model = load_model_cached(str(weights_path), model_name)
except Exception as e:
    st.error(f"Failed to load model checkpoint for '{model_name}': {e}")
    st.stop()

rgb = to_rgb_for_display(x13, rgb_idxs)

t0 = time.perf_counter()
x_t = torch.from_numpy(x13).unsqueeze(0).to(device)
x_norm = normalize_batch(x_t, mean_13, std_13)

with torch.no_grad():
    logits = model(x_norm)
    probs = torch.sigmoid(logits)
    pred = (probs >= t_best).float()

t1 = time.perf_counter()
infer_ms = (t1 - t0) * 1000.0

logits_np = logits[0, 0].detach().cpu().numpy()
probs_np = probs[0, 0].detach().cpu().numpy()
pred_np = pred[0, 0].detach().cpu().numpy()

overlay = overlay_mask(rgb, pred_np, alpha=0.35)
prob_heat = prob_to_heatmap_gray(probs_np)

pred_fire_pixels = int(pred_np.sum())
total_pixels = int(pred_np.size)
pred_fire_ratio = float(pred_fire_pixels / max(total_pixels, 1))

# =========================
# PIPELINE INTERNALS
# =========================
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("## Pipeline Internals (Numbers)")

colA, colB = st.columns([1.05, 1.0], gap="large")

with colA:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 1) Load & Validate")
    st.markdown(
        f"""
<div style="display:grid; grid-template-columns: 220px 1fr; gap: 6px 10px; font-size:0.95rem;">
  <div style="color:rgba(255,255,255,0.65);">Selected model</div><div style="font-family:ui-monospace; color:rgba(255,255,255,0.95);">{model_name}</div>
  <div style="color:rgba(255,255,255,0.65);">Uploaded file</div><div style="font-family:ui-monospace; color:rgba(255,255,255,0.95);">{uploaded.name}</div>
  <div style="color:rgba(255,255,255,0.65);">Required keys</div><div style="font-family:ui-monospace; color:rgba(255,255,255,0.95);">image, aerosol</div>
  <div style="color:rgba(255,255,255,0.65);">image shape</div><div style="font-family:ui-monospace; color:rgba(255,255,255,0.95);">{tuple(x13[:12].shape)}</div>
  <div style="color:rgba(255,255,255,0.65);">aerosol shape</div><div style="font-family:ui-monospace; color:rgba(255,255,255,0.95);">{tuple(x13[12].shape)}</div>
  <div style="color:rgba(255,255,255,0.65);">model input x shape</div><div style="font-family:ui-monospace; color:rgba(255,255,255,0.95);">{tuple(x13.shape)}</div>
  <div style="color:rgba(255,255,255,0.65);">512×512 expected</div><div style="font-family:ui-monospace; color:rgba(255,255,255,0.95);">{'✅' if size_ok else f'⚠️ got {H}×{W}'}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
    st.markdown("### 2) Segmentation Output (Logits)")
    st.markdown(
        f"""
<div style="display:grid; grid-template-columns: 220px 1fr; gap: 6px 10px; font-size:0.95rem;">
  <div style="color:rgba(255,255,255,0.65);">logits shape</div><div style="font-family:ui-monospace; color:rgba(255,255,255,0.95);">{tuple(logits.shape)}</div>
  <div style="color:rgba(255,255,255,0.65);">logits min / max</div><div style="font-family:ui-monospace; color:rgba(255,255,255,0.95);">{float(logits_np.min()):.4f} / {float(logits_np.max()):.4f}</div>
  <div style="color:rgba(255,255,255,0.65);">logits mean / std</div><div style="font-family:ui-monospace; color:rgba(255,255,255,0.95);">{float(logits_np.mean()):.4f} / {float(logits_np.std()):.4f}</div>
  <div style="color:rgba(255,255,255,0.65);">inference time</div><div style="font-family:ui-monospace; color:rgba(255,255,255,0.95);">{infer_ms:.2f} ms</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with colB:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 3) Sigmoid (Probabilities)")
    st.markdown(
        f"""
<div style="display:grid; grid-template-columns: 220px 1fr; gap: 6px 10px; font-size:0.95rem;">
  <div style="color:rgba(255,255,255,0.65);">probs shape</div><div style="font-family:ui-monospace; color:rgba(255,255,255,0.95);">{tuple(probs.shape)}</div>
  <div style="color:rgba(255,255,255,0.65);">probs min / max</div><div style="font-family:ui-monospace; color:rgba(255,255,255,0.95);">{float(probs_np.min()):.4f} / {float(probs_np.max()):.4f}</div>
  <div style="color:rgba(255,255,255,0.65);">probs mean / std</div><div style="font-family:ui-monospace; color:rgba(255,255,255,0.95);">{float(probs_np.mean()):.4f} / {float(probs_np.std()):.4f}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
    st.markdown("### 4) Data-driven Thresholding (Validation-derived)")
    st.markdown(
        f"""
<div style="color:rgba(255,255,255,0.70); font-size:0.95rem; line-height:1.35;">
Threshold <code>t_best</code> is taken from a validation threshold sweep (data-driven). It is then applied to the probability map.
</div>
<div style="display:grid; grid-template-columns: 220px 1fr; gap: 6px 10px; font-size:0.95rem; margin-top:10px;">
  <div style="color:rgba(255,255,255,0.65);">t_best</div><div style="font-family:ui-monospace; color:rgba(255,255,255,0.95);">{t_best}</div>
  <div style="color:rgba(255,255,255,0.65);">pred fire pixels</div><div style="font-family:ui-monospace; color:rgba(255,255,255,0.95);">{pred_fire_pixels} / {total_pixels}</div>
  <div style="color:rgba(255,255,255,0.65);">pred fire ratio</div><div style="font-family:ui-monospace; color:rgba(255,255,255,0.95);">{pred_fire_ratio:.4f}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# VISUALIZATION & PRODUCT
# =========================
st.markdown("<hr/>", unsafe_allow_html=True)
st.markdown("## Visualization & Product")

v1, v2, v3, v4 = st.columns(4, gap="medium")

with v1:
    st.markdown("**Original (RGB composite)**")
    st.image((rgb * 255).astype(np.uint8), use_column_width=True)

with v2:
    st.markdown("**Probability (sigmoid)**")
    st.image((prob_heat * 255).astype(np.uint8), use_column_width=True)

with v3:
    st.markdown("**Binary mask (probs ≥ t_best)**")
    mask_vis = np.stack([pred_np, pred_np, pred_np], axis=-1)
    st.image((mask_vis * 255).astype(np.uint8), use_column_width=True)

with v4:
    st.markdown("**Predicted fire overlay (product)**")
    st.image((overlay * 255).astype(np.uint8), use_column_width=True)

st.markdown(
    """
<div style="color:rgba(255,255,255,0.65); font-size:0.90rem; margin-top:10px;">
Notes: This app runs pixel-wise inference only. Any patch-level aggregation or risk mapping is intentionally removed to match the latest methodology.
</div>
""",
    unsafe_allow_html=True,
)
