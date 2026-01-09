# app7.py
# Sen2Fire — Pixel-wise Segmentation (Tabs)
# Tab 1: Single Model Inference (Pipeline Internals + Metrics Summary + Test Metrics hard-coded)
# Tab 2: Model Comparison (visual only)
#
# Pipeline: logits -> sigmoid -> threshold (validation-derived) -> visualization/product
# Patch-level logic removed.

import io
import time
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import streamlit as st
import segmentation_models_pytorch as smp

# Optional (PR AUC / Average Precision)
try:
    from sklearn.metrics import average_precision_score
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# ============================================================
# METRICS (PATCH-LEVEL)
# ============================================================
def compute_patch_metrics(gt01: np.ndarray, probs_np: np.ndarray, mask_np: np.ndarray) -> dict:
    """
    Pixel-wise metrics on one patch.
    gt01: (H,W) {0,1}
    probs_np: (H,W) [0,1]
    mask_np: (H,W) {0,1}
    """
    gt = (gt01 > 0.5).astype(np.uint8).ravel()
    pr = np.clip(probs_np.astype(np.float32), 1e-7, 1 - 1e-7).ravel()
    pd_ = (mask_np > 0.5).astype(np.uint8).ravel()

    tp = int(((pd_ == 1) & (gt == 1)).sum())
    fp = int(((pd_ == 1) & (gt == 0)).sum())
    fn = int(((pd_ == 0) & (gt == 1)).sum())
    tn = int(((pd_ == 0) & (gt == 0)).sum())
    total = tp + fp + fn + tn

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = (2 * precision * recall) / max(precision + recall, 1e-12)
    acc       = (tp + tn) / max(total, 1)

    # BCE on probabilities
    bce = float(-(gt * np.log(pr) + (1 - gt) * np.log(1 - pr)).mean())

    pr_auc = None
    if SKLEARN_OK:
        if (gt.sum() > 0) and (gt.sum() < gt.size):
            pr_auc = float(average_precision_score(gt, pr))

    return {
        "loss": bce,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "acc": float(acc),
        "pr_auc": pr_auc,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }


# ============================================================
# NORMALIZATION STATS
# ============================================================
def load_norm_stats(path: Path):
    """
    Expected json keys:
      {
        "mean_13": [...13...],
        "std_13":  [...13...]
      }

    Optional keys supported:
      "scale_div": 10000.0
      "clip": [0.0, 1.0]
      "apply_scale_clip": true/false
    """
    with open(path, "r") as f:
        d = json.load(f)

    mean_13 = d["mean_13"]
    std_13  = d["std_13"]

    # optional (default OFF to match notebook-style)
    apply_scale_clip = bool(d.get("apply_scale_clip", False))
    scale_div = d.get("scale_div", None)
    clip = d.get("clip", None)

    return mean_13, std_13, apply_scale_clip, scale_div, clip


# ============================================================
# PATHS + MODELS
# ============================================================
APP_DIR = Path(__file__).resolve().parent

# NOTE: "DeepLabV3+ (EfficientNet-B4) — non-pretrained" REMOVED as requested.
WEIGHTS = {
    "U-Net (JP retrained)": APP_DIR / "thesis_unet" / "unet_best_retrained2.pth",
    "U-Net Eff (EfficientNet-B4)": APP_DIR / "unet_eff" / "efficientb4_pretrained.pth",
    "DeepLabV3+ (EfficientNet-B4) — pretrained": APP_DIR / "thesis_deeplab" / "deeplab_pretrained.pth",
}

METRICS = {
    "U-Net (JP retrained)": [
        APP_DIR / "thesis_unet" / "training_metrics1.xlsx",
        APP_DIR / "thesis_unet" / "training_metrics2.xlsx",
    ],
    "U-Net Eff (EfficientNet-B4)": [
        APP_DIR / "unet_eff" / "training_metrics_eff_pretrained.xlsx",
    ],
    "DeepLabV3+ (EfficientNet-B4) — pretrained": [
        APP_DIR / "thesis_deeplab" / "pretraining_metrics_deeplab.xlsx",
    ],
}

# Thresholds (as shown in your desired labels on image 2 -> all 0.05)
T_BEST = {
    "U-Net (JP retrained)": 0.15,
    "U-Net Eff (EfficientNet-B4)": 0.05,
    "DeepLabV3+ (EfficientNet-B4) — pretrained": 0.05,
}

# IMPORTANT:
# norm_stats.json harus berasal dari training pipeline yang sama.
# Default-nya: hanya mean/std, tanpa scale_div/clip.
PREPROCESS = {
    "U-Net (JP retrained)": {
        "norm_path": APP_DIR / "thesis_unet" / "norm_stats.json",
    },
    "U-Net Eff (EfficientNet-B4)": {
        "norm_path": APP_DIR / "unet_eff" / "norm_stats.json",
    },
    "DeepLabV3+ (EfficientNet-B4) — pretrained": {
        "norm_path": APP_DIR / "thesis_deeplab" / "norm_stats.json",
    },
}

ARCH = {
    "U-Net (JP retrained)": ("unet", "resnet18"),
    "U-Net Eff (EfficientNet-B4)": ("unet", "efficientnet-b4"),
    "DeepLabV3+ (EfficientNet-B4) — pretrained": ("deeplab", "efficientnet-b4"),
}

TEST_METRICS = {
    "DeepLabV3+ (EfficientNet-B4) — pretrained": {
        "test_loss": 0.0682,
        "test_precision": 0.6771,
        "test_recall": 0.6223,
        "test_f1": 0.6485,
        "test_acc": 0.9836,
        "test_pr_auc": 0.7013,
    },
}

RGB_IDXS = (0, 1, 2)

# DISPLAY NAMES (match your image-2 naming)
DISPLAY_NAME = {
    "U-Net (JP retrained)": "U-Net + Resnet18 (thr: 0.15)",
    "U-Net Eff (EfficientNet-B4)": "U-Net + EfficientNet-b4 (thr: 0.05)",
    "DeepLabV3+ (EfficientNet-B4) — pretrained": "DeepLabV3Plus + EfficientNet-b4 (thr: 0.05)",
}


# ============================================================
# HELPERS
# ============================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_batch(x: torch.Tensor, mean_13, std_13) -> torch.Tensor:
    mean = torch.tensor(mean_13, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std  = torch.tensor(std_13,  device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - mean) / (std + 1e-6)


def build_model(model_key: str):
    arch, enc = ARCH[model_key]
    if arch == "unet":
        return smp.Unet(
            encoder_name=enc,
            encoder_weights="imagenet",
            in_channels=13,
            classes=1,
            activation=None,
        )
    if arch == "deeplab":
        return smp.DeepLabV3Plus(
            encoder_name=enc,
            encoder_weights="imagenet",
            in_channels=13,
            classes=1,
            activation=None,
        )
    raise ValueError(f"Unsupported arch: {arch}")


@st.cache_resource
def load_model_cached(model_key: str):
    weights_path = WEIGHTS[model_key]
    device = get_device()
    model = build_model(model_key).to(device)

    load_kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False

    ckpt = torch.load(str(weights_path), **load_kwargs)
    state = ckpt.get("model_state", ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_npz_from_bytes(npz_bytes: bytes):
    """
    Returns:
      x13: (13,H,W) float32
      gt01: (H,W) float32 in {0,1} if 'label' exists, else None
    """
    with np.load(io.BytesIO(npz_bytes)) as data:
        keys = list(data.files)
        if "image" not in keys or "aerosol" not in keys:
            raise KeyError(f"NPZ missing required keys. Found keys: {keys}")

        img = data["image"].astype(np.float32)      # (12,H,W)
        aer = data["aerosol"].astype(np.float32)    # (H,W)
        gt  = data["label"] if "label" in keys else None

    if img.ndim != 3:
        raise ValueError(f"`image` must be 3D (C,H,W). Got {img.shape}")
    if aer.ndim != 2:
        raise ValueError(f"`aerosol` must be 2D (H,W). Got {aer.shape}")

    x13 = np.concatenate([img, aer[None, ...]], axis=0)  # (13,H,W)

    gt01 = None
    if gt is not None:
        gt = gt.astype(np.float32)
        if gt.ndim != 2:
            raise ValueError(f"`label` must be 2D (H,W). Got {gt.shape}")
        gt01 = (gt > 0.5).astype(np.float32)

    return x13, gt01


def to_rgb_for_display(x13: np.ndarray, rgb_idxs=RGB_IDXS) -> np.ndarray:
    rgb = np.stack([x13[rgb_idxs[0]], x13[rgb_idxs[1]], x13[rgb_idxs[2]]], axis=-1)
    out = np.zeros_like(rgb, dtype=np.float32)
    for c in range(3):
        ch = rgb[..., c]
        lo, hi = np.percentile(ch, 2), np.percentile(ch, 98)
        out[..., c] = np.clip((ch - lo) / (hi - lo + 1e-6), 0, 1)
    return out


def overlay_binary(rgb01: np.ndarray, mask01: np.ndarray, alpha=0.4) -> np.ndarray:
    rgb = rgb01.copy()
    red = np.zeros_like(rgb)
    red[..., 0] = 1.0
    m = (mask01 > 0.5).astype(np.float32)[..., None]
    return np.clip((1 - alpha * m) * rgb + (alpha * m) * red, 0, 1)


def overlay_prob_red(rgb01: np.ndarray, prob01: np.ndarray, alpha=0.4) -> np.ndarray:
    """
    Notebook-style:
      imshow(img_vis)
      imshow(prob, cmap='Reds', alpha=0.4)

    Approx with RGB blending, intensity follows prob.
    """
    p = np.clip(prob01.astype(np.float32), 0, 1)[..., None]
    red = np.zeros_like(rgb01)
    red[..., 0] = 1.0
    return np.clip((1 - alpha * p) * rgb01 + (alpha * p) * red, 0, 1)


def red_mask_on_white(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 > 0.5).astype(np.float32)
    out = np.ones((m.shape[0], m.shape[1], 3), dtype=np.float32)
    out[..., 1] = 1.0 - 0.85 * m
    out[..., 2] = 1.0 - 0.85 * m
    return out


def bw_binary_mask(mask01: np.ndarray) -> np.ndarray:
    """
    Black-white mask:
      - 0 -> black
      - 1 -> white
    Returned as 3-channel float in [0,1] so st.image looks consistent.
    """
    m = (mask01 > 0.5).astype(np.float32)
    return np.stack([m, m, m], axis=-1)


def stats_np(arr: np.ndarray) -> dict:
    return {
        "shape": str(tuple(arr.shape)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def stats_torch(t: torch.Tensor) -> dict:
    tt = t.detach().float().cpu()
    return {
        "shape": str(tuple(tt.shape)),
        "min": float(tt.min().item()),
        "max": float(tt.max().item()),
        "mean": float(tt.mean().item()),
        "std": float(tt.std().item()),
    }


def infer_with_internals(model_key: str, x13: np.ndarray):
    """
    Returns:
      visuals: rgb, pred_prob_overlay, pred_bin_overlay, pred_bin_bw
      internals: dict of stats
      raw arrays: probs_np, mask_np
    """
    device = get_device()
    model = load_model_cached(model_key)

    pp = PREPROCESS[model_key]
    mean_13, std_13, apply_scale_clip, scale_div, clip = load_norm_stats(pp["norm_path"])

    # Match notebook-style first (default): only normalize_batch(mean/std)
    x13p = x13.astype(np.float32)

    # Optional: ONLY if your norm_stats.json explicitly says so
    if apply_scale_clip:
        if scale_div is not None:
            x13p = x13p / float(scale_div)
        if clip is not None and isinstance(clip, (list, tuple)) and len(clip) == 2:
            lo, hi = float(clip[0]), float(clip[1])
            x13p = np.clip(x13p, lo, hi)

    x_raw = torch.from_numpy(x13p).unsqueeze(0).to(device)   # (1,13,H,W)
    x_norm = normalize_batch(x_raw, mean_13, std_13)

    # threshold slider (debug)
    t_default = float(T_BEST[model_key])
    # t_used = st.slider(
    #     "Threshold (t_best) — debug override",
    #     min_value=0.00, max_value=1.00,
    #     value=t_default,
    #     step=0.01
    # )

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = model(x_norm)                 # (1,1,H,W)
        probs  = torch.sigmoid(logits)         # (1,1,H,W)
        mask   = (probs >= t_default).float()     # (1,1,H,W)
    t1 = time.perf_counter()

    logits_np = logits[0, 0].detach().cpu().numpy()
    probs_np  = probs[0, 0].detach().cpu().numpy()
    mask_np   = mask[0, 0].detach().cpu().numpy()

    pred_fire_pixels = int(mask_np.sum())
    total_pixels = int(mask_np.size)
    pred_fire_ratio = float(pred_fire_pixels / max(total_pixels, 1))

    # Display RGB from ORIGINAL x13 (not normalized)
    rgb = to_rgb_for_display(x13)

    # Notebook-style visuals
    pred_prob_overlay = overlay_prob_red(rgb, probs_np, alpha=0.4)
    pred_bin_overlay  = overlay_binary(rgb, mask_np, alpha=0.4)

    # NEW: B/W binary mask for your requested "gambar ke-3 jadi hitam putih"
    pred_bin_bw = bw_binary_mask(mask_np)

    internals = {
        "device": str(device),
        "t_used": float(t_default),
        "infer_s": float(t1 - t0),
        "apply_scale_clip": bool(apply_scale_clip),
        "scale_div": scale_div,
        "clip": clip,
        "x13_raw_tensor": stats_torch(x_raw),
        "x13_norm_tensor": stats_torch(x_norm),
        "logits": stats_np(logits_np),
        "probs": stats_np(probs_np),
        "mask": {
            "shape": str(mask_np.shape),
            "fire_pixels": pred_fire_pixels,
            "total_pixels": total_pixels,
            "fire_ratio": pred_fire_ratio,
        },
    }

    visuals = {
        "rgb": rgb,
        "pred_prob_overlay": pred_prob_overlay,
        "pred_bin_overlay": pred_bin_overlay,
        "pred_bin_bw": pred_bin_bw,
    }

    return visuals, internals, probs_np, mask_np


@st.cache_data
def load_metrics_df(model_key: str) -> pd.DataFrame:
    paths = METRICS[model_key]
    frames = []
    for i, p in enumerate(paths, start=1):
        if not p.exists():
            continue
        df = pd.read_excel(p)
        df["run"] = f"run{i}"
        df["source_file"] = p.name
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out.columns = [c.strip() for c in out.columns]
    return out


def metrics_best_rows(df: pd.DataFrame):
    if df.empty or "va_f1" not in df.columns:
        return None, pd.DataFrame()

    best_per_run = (
        df.sort_values("va_f1", ascending=False)
          .groupby("run", as_index=False)
          .head(1)
          .sort_values("run")
    )

    best_overall = df.loc[df["va_f1"].idxmax()]
    return best_overall, best_per_run


def pick_auc_column(df: pd.DataFrame) -> str | None:
    for c in ["va_PRauc", "va_prauc", "va_pr_auc", "va_auc", "va_AUC"]:
        if c in df.columns:
            return c
    return None


# ============================================================
# UI STYLING
# ============================================================
st.set_page_config(page_title="Sen2Fire", layout="wide")

st.markdown(
    """
<style>
html, body, [class*="css"]  { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
.block-container { padding-top: 1.2rem; }
.center-wrap { max-width: 980px; margin: 0 auto; }
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
.card {
    border: 1px solid rgba(255,255,255,0.10);
    background: rgba(255,255,255,0.03);
    border-radius: 16px;
    padding: 14px 14px 10px 14px;
}
hr { border: none; border-top: 1px solid rgba(255,255,255,0.10); margin: 18px 0; }
.small { color: rgba(255,255,255,0.70); font-size: 0.92rem; line-height: 1.35; }
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HEADER
# ============================================================
st.title("Sen2Fire — Pixel-wise Segmentation")
st.markdown("Segmentation (logits) → sigmoid → data-driven threshold (validation-derived) → visualization & product.")

# validate paths early
missing = []
for k, p in WEIGHTS.items():
    if not p.exists():
        missing.append(f"{k} weights missing: {p}")
for k, d in PREPROCESS.items():
    npth = d["norm_path"]
    if not npth.exists():
        missing.append(f"{k} norm_stats.json missing: {npth}")

if missing:
    st.error("Missing files:\n\n" + "\n".join(missing))
    st.stop()

tab_single, tab_compare = st.tabs(["Single Model Inference", "Model Comparison"])


# ============================================================
# TAB 1 — SINGLE
# ============================================================
with tab_single:
    st.markdown('<div class="center-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    st.markdown("## Input")

    c1, c2, c3 = st.columns([1, 2.2, 1])
    with c2:
        model_key = st.selectbox(
            "Choose model",
            list(WEIGHTS.keys()),
            index=0,
            label_visibility="collapsed",
            format_func=lambda k: DISPLAY_NAME.get(k, k),
        )

    u1, u2, u3 = st.columns([1, 2.2, 1])
    with u2:
        uploaded = st.file_uploader(
            "Upload test patch (.npz)",
            type=["npz"],
            key="single_upload",
            label_visibility="collapsed"
        )

    st.markdown(
        """
<div class="center-muted">
Accepted: <b>.npz</b> • Recommended patch size: <b>512×512</b><br/>
Required keys: <code>image</code>, <code>aerosol</code> • Optional: <code>label</code> (for Ground Truth display)
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)  # upload-card

    with st.expander("Input specification (what this app expects)"):
        st.markdown(
            """
- File format: `.npz` (NumPy zipped arrays)
- Required arrays:
  - `image`: `(12, H, W)`
  - `aerosol`: `(H, W)`
- Optional arrays:
  - `label`: `(H, W)` (Ground Truth mask; used only for visualization & patch metrics)
- Model input: concatenation → `(13, H, W)`
- Output: logits → sigmoid probabilities (0–1) → threshold `t_best` → binary mask
            """.strip()
        )
    st.markdown("</div>", unsafe_allow_html=True)  # center-wrap

    if uploaded is None:
        st.info("Choose a model, then upload a .npz file to start.")
        st.stop()

    try:
        x13, gt01 = load_npz_from_bytes(uploaded.getvalue())
    except Exception as e:
        st.error(f"Failed to read NPZ: {e}")
        st.stop()

    H, W = x13.shape[1], x13.shape[2]
    size_ok = (H == 512 and W == 512)

    visuals, internals, probs_np, mask_np = infer_with_internals(model_key, x13)

    # st.markdown("### Debug: distribution")
    # st.json({
    #     "model_display": DISPLAY_NAME.get(model_key, model_key),
    #     "t_used": internals["t_used"],
    #     "apply_scale_clip": internals["apply_scale_clip"],
    #     "scale_div": internals["scale_div"],
    #     "clip": internals["clip"],
    #     "probs_min": float(probs_np.min()),
    #     "probs_max": float(probs_np.max()),
    #     "probs_mean": float(probs_np.mean()),
    #     "mask_sum_pixels": int(mask_np.sum()),
    #     "mask_ratio": float(mask_np.mean()),
    #     "gt_sum_pixels": None if gt01 is None else int(gt01.sum()),
    #     "gt_ratio": None if gt01 is None else float(gt01.mean()),
    # })

    # ========================================================
    # MODEL INFO
    # ========================================================
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("## Model Info")
    st.markdown(f"""
**Model Name:**  
{DISPLAY_NAME.get(model_key, model_key)}

**Threshold used (t_used):**  
{internals['t_used']}

**Patch Size:**  
{H} × {W}{" ✅" if size_ok else " ⚠️"}

**Input:**  
Concatenation → **(13, H, W)** (12-band + aerosol)

**Output:**  
Logits → sigmoid → threshold → binary mask
""")

    # ========================================================
    # QUICK SUMMARY / PATCH METRICS
    # ========================================================
    st.markdown("<hr/>", unsafe_allow_html=True)
    if gt01 is not None:
        st.markdown("## Selected Patch Prediction Overview")
        st.markdown('<div class="small">metrics: loss, precision, recall, f1, accuracy, PR_AUC</div>', unsafe_allow_html=True)

        pm = compute_patch_metrics(gt01, probs_np, mask_np)
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Loss (BCE)", f"{pm['loss']:.4f}")
        c2.metric("Precision", f"{pm['precision']:.4f}")
        c3.metric("Recall", f"{pm['recall']:.4f}")
        c4.metric("F1-score", f"{pm['f1']:.4f}")
        c5.metric("Accuracy", f"{pm['acc']:.4f}")
        c6.metric("PR_AUC", "N/A" if pm["pr_auc"] is None else f"{pm['pr_auc']:.4f}")

        with st.expander("Confusion details (TP/FP/FN/TN)"):
            st.json({k: pm[k] for k in ["tp", "fp", "fn", "tn"]})
    else:
        st.info("No Ground Truth `label` found → patch metrics not computed.")

    # ========================================================
    # PIPELINE INTERNALS
    # ========================================================
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("## Pipeline Internals (Numbers)")
    left, right = st.columns([1.05, 1.0], gap="large")

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### A) Input Tensors")
        st.markdown('<div class="small">Raw tensor & normalized tensor fed to the model.</div>', unsafe_allow_html=True)
        st.markdown("**x13_raw_tensor**")
        st.json(internals["x13_raw_tensor"])
        st.markdown("**x13_norm_tensor**")
        st.json(internals["x13_norm_tensor"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
        st.markdown("### B) Segmentation Output (Logits)")
        st.json(internals["logits"])
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### C) Sigmoid Probabilities")
        st.json(internals["probs"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
        st.markdown("### D) Thresholding Output (Binary Mask)")
        st.json(internals["mask"])
        st.markdown("</div>", unsafe_allow_html=True)

    # ========================================================
    # VISUALIZATION & PRODUCT
    # ========================================================
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("## Visualization & Product (Notebook-style)")

    v1, v2, v3, v4 = st.columns(4, gap="medium")

    v1.image((visuals["rgb"] * 255).astype(np.uint8), caption="Original (RGB composite)", use_column_width=True)

    if gt01 is None:
        v2.warning("Ground Truth not found (missing key: label)")
    else:
        gt_vis = red_mask_on_white(gt01)
        gt_ratio = float(gt01.mean())
        v2.image((gt_vis * 255).astype(np.uint8), caption=f"Ground Truth (fire ratio={gt_ratio:.4f})", use_column_width=True)

    # CHANGED: slot #3 now B/W binary mask (as you requested)
    v3.image((visuals["pred_bin_bw"] * 255).astype(np.uint8),
             caption="Prediction (binary mask, B/W)",
             use_column_width=True)

    v4.image((visuals["pred_bin_overlay"] * 255).astype(np.uint8),
             caption="Prediction (binary overlay / product)",
             use_column_width=True)

    # ========================================================
    # METRICS SUMMARY
    # ========================================================
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("## Model Quick Summary")

    dfm = load_metrics_df(model_key)
    if dfm.empty:
        st.warning("No metrics file found/readable for this model.")
    else:
        auc_col = pick_auc_column(dfm)
        best_overall, best_per_run = metrics_best_rows(dfm)

        if best_overall is None or best_per_run.empty:
            st.warning("Metrics file loaded, but required column `va_f1` was not found.")
        else:
            cA, cB, cC, cD = st.columns(4)
            cA.metric("Best overall run", str(best_overall.get("run", "")))
            cB.metric("Best overall epoch", str(int(best_overall.get("epoch", -1))))
            cC.metric("Best overall va_f1", f"{float(best_overall.get('va_f1', 0.0)):.4f}")
            if auc_col:
                cD.metric("Below are the evaluation results of the model retrained with an adjusted threshold.")
            else:
                cD.metric("va_pr_auc", "N/A")

            cols_to_show = ["run", "epoch", "va_loss", "va_precision", "va_recall", "va_f1"]
            if auc_col:
                cols_to_show.append(auc_col)
            if "seconds" in dfm.columns:
                cols_to_show.append("seconds")
            cols_to_show = [c for c in cols_to_show if c in dfm.columns]

            st.markdown("### Best epoch per run (va_f1)")
            st.dataframe(
                best_per_run[cols_to_show].rename(columns={auc_col: "va_pr_auc"} if auc_col else {}),
                use_container_width=True
            )

            st.markdown("### Top-5 overall epochs (va_f1)")
            top5 = dfm.sort_values("va_f1", ascending=False).head(5)
            st.dataframe(
                top5[cols_to_show].rename(columns={auc_col: "va_pr_auc"} if auc_col else {}),
                use_container_width=True
            )

    # ========================================================
    # TEST METRICS (HARD-CODED)
    # ========================================================
    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("### Test Metrics (hard-coded)")

    tm = TEST_METRICS.get(model_key)
    if tm is None:
        st.info("Test metrics for this model are not provided yet.")
    else:
        df_test = pd.DataFrame([{
            "test_loss": tm.get("test_loss"),
            "test_precision": tm.get("test_precision"),
            "test_recall": tm.get("test_recall"),
            "test_f1": tm.get("test_f1"),
            "test_acc": tm.get("test_acc"),
            "test_pr_auc": tm.get("test_pr_auc"),
        }])
        st.dataframe(df_test, use_container_width=True)

    st.markdown(
        '<div class="small" style="margin-top:10px;">Notes: This app is pixel-wise only. Patch-level aggregation is removed.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# TAB 2 — COMPARISON (visual only)
# ============================================================
with tab_compare:
    st.subheader("Model Comparison (Visual Only)")

    models = st.multiselect(
        "Select models to compare",
        list(WEIGHTS.keys()),
        default=list(WEIGHTS.keys())[:2],
        key="compare_models",
        format_func=lambda k: DISPLAY_NAME.get(k, k),
    )

    uploaded2 = st.file_uploader(
        "Upload test patch (.npz)",
        type=["npz"],
        key="compare_upload",
    )

    if uploaded2 is None:
        st.info("Choose models, then upload a .npz file to compare overlays.")
        st.stop()

    if not models:
        st.warning("Select at least one model.")
        st.stop()

    try:
        x13c, _gt_unused = load_npz_from_bytes(uploaded2.getvalue())
    except Exception as e:
        st.error(f"Failed to read NPZ: {e}")
        st.stop()

    rgb = to_rgb_for_display(x13c)

    st.markdown("### Overlays (same input, different models)")
    cols = st.columns(len(models) + 1, gap="medium")
    cols[0].image((rgb * 255).astype(np.uint8), caption="RGB", use_column_width=True)

    for i, m in enumerate(models):
        visuals_m, _, _, _ = infer_with_internals(m, x13c)
        cols[i + 1].image(
            (visuals_m["pred_bin_overlay"] * 255).astype(np.uint8),
            caption=DISPLAY_NAME.get(m, m),
            use_column_width=True
        )
