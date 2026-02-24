import io
import time
from pathlib import Path
import inspect

import numpy as np
import torch
import streamlit as st

# pip install segmentation-models-pytorch
import segmentation_models_pytorch as smp


# =========================
# USER CONFIG (EDIT THESE)
# =========================
WEIGHTS_PATH = r"D:\BINUS\Thesis\Zenodo_Sen2Fire\Streamlit_Initial\thesis_unet\unet_best_retrained2.pth"

# PLACEBO for now (len=13 each). Replace later with JP's printed values.
MEAN_13 = [0.0] * 13
STD_13  = [1.0] * 13

# Fixed best threshold from JP
T_BEST = 0.15

# Fixed RGB composite mapping
RGB_IDXS = (0, 1, 2)


# =========================
# CORE HELPERS
# =========================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize_batch(x: torch.Tensor, mean_13, std_13) -> torch.Tensor:
    mean = torch.tensor(mean_13, device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    std  = torch.tensor(std_13,  device=x.device, dtype=x.dtype).view(1, -1, 1, 1)
    return (x - mean) / (std + 1e-6)


def build_unet(in_channels=13):
    return smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=1,
        activation=None,  # logits
    )


@st.cache_resource
def load_model(weights_path: str):
    device = get_device()
    model = build_unet(in_channels=13).to(device)

    load_kwargs = {"map_location": device}
    sig = inspect.signature(torch.load)
    if "weights_only" in sig.parameters:
        load_kwargs["weights_only"] = False

    ckpt = torch.load(weights_path, **load_kwargs)

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
    bio = io.BytesIO(npz_bytes)
    with np.load(bio) as data:
        if "image" not in data.files or "aerosol" not in data.files:
            raise KeyError(f"NPZ missing required keys. Found: {data.files}")

        img = data["image"].astype(np.float32)
        aer = data["aerosol"].astype(np.float32)
        y = data["label"].astype(np.float32) if "label" in data.files else None

    aer = aer[None, ...]
    x13 = np.concatenate([img, aer], axis=0)
    return x13, y


def stats_np(arr: np.ndarray):
    return {
        "shape": tuple(arr.shape),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
    }


def stats_torch(t: torch.Tensor):
    t2 = t.detach().float().cpu()
    return {
        "shape": tuple(t2.shape),
        "min": float(t2.min().item()),
        "max": float(t2.max().item()),
        "mean": float(t2.mean().item()),
        "std": float(t2.std().item()),
    }


def to_rgb_for_display(x13: np.ndarray, rgb_idxs=RGB_IDXS):
    rgb = np.stack(
        [x13[rgb_idxs[0]], x13[rgb_idxs[1]], x13[rgb_idxs[2]]],
        axis=-1
    )

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


# =========================
# STREAMLIT UI
# =========================
st.set_page_config(page_title="Sen2Fire U-Net Inference", layout="wide")
st.title("Sen2Fire U-Net — Transparent Inference Pipeline")
st.caption("Upload 1 test patch (.npz) → show internals → predicted fire overlay (fixed t_best=0.15, fixed RGB).")

if len(MEAN_13) != 13 or len(STD_13) != 13:
    st.error("MEAN_13 and STD_13 must each have length 13.")
    st.stop()

weights_path = Path(WEIGHTS_PATH)
if not weights_path.exists():
    st.error(f"WEIGHTS_PATH not found: {WEIGHTS_PATH}")
    st.stop()

if MEAN_13 == [0.0] * 13 and STD_13 == [1.0] * 13:
    st.warning(
        "Normalization is currently PLACEBO (mean=0, std=1). "
        "Results are for pipeline validation only."
    )

st.sidebar.header("Input")
uploaded = st.sidebar.file_uploader("Upload 1 test .npz patch", type=["npz"])
st.sidebar.write("Fixed threshold (t_best):", T_BEST)
st.sidebar.write("Fixed RGB channels:", RGB_IDXS)

if uploaded is None:
    st.info("Upload a .npz file to start.")
    st.stop()

x13, y = load_npz_from_bytes(uploaded.getvalue())
rgb = to_rgb_for_display(x13, RGB_IDXS)

device = get_device()
model = load_model(str(weights_path))

t0 = time.perf_counter()
x_t = torch.from_numpy(x13).unsqueeze(0).to(device)
x_norm = normalize_batch(x_t, MEAN_13, STD_13)

with torch.no_grad():
    logits = model(x_norm)
    probs = torch.sigmoid(logits)
    pred = (probs >= T_BEST).float()

t1 = time.perf_counter()
infer_ms = (t1 - t0) * 1000.0

probs_np = probs[0, 0].cpu().numpy()
pred_np = pred[0, 0].cpu().numpy()
overlay = overlay_mask(rgb, pred_np)

gt_np = (y > 0.5).astype(np.float32) if y is not None else None


# =========================
# PIPELINE NUMBERS
# =========================
st.subheader("Pipeline Internals (Numbers)")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.write("Uploaded file:", uploaded.name)
    st.write("Device:", str(device))
    st.write("Inference time (ms):", f"{infer_ms:.2f}")

with c2:
    st.json({"x13_raw": stats_np(x13), "label_raw": stats_np(y) if y is not None else None})

with c3:
    st.json({"x_norm": stats_torch(x_norm)})

with c4:
    st.json({
        "logits": stats_torch(logits),
        "probs": stats_torch(probs),
        "t_best": T_BEST,
        "pred_fire_ratio": float(pred_np.mean()),
        "pred_fire_pixels": int(pred_np.sum()),
        "total_pixels": int(pred_np.size),
    })


# =========================
# PATCH-LEVEL INTERPRETATION
# =========================
area = float(pred_np.mean())
a1, a2 = 0.15, 0.30

if area <= 1e-3:
    cat = "NO_FIRE"
elif area < a1:
    cat = "LOW_FIRE"
elif area < a2:
    cat = "MID_FIRE"
else:
    cat = "HIGH_FIRE"

st.subheader("Patch-level Interpretation")
st.write(f"Predicted fire area ratio = {area:.4f}")
st.write(f"Category (a1={a1}, a2={a2}) → **{cat}**")
if gt_np is not None:
    st.write(f"Ground-truth fire area ratio = {float(gt_np.mean()):.4f}")


# =========================
# VISUALS
# =========================
st.subheader("Visuals")
left, mid, right = st.columns(3)

with left:
    st.markdown("**Original Image (fixed RGB composite)**")
    st.image((rgb * 255).astype(np.uint8), use_column_width=True)

with mid:
    st.markdown("**Ground Truth Fire**")
    if gt_np is None:
        st.info("No `label` found in this .npz file.")
    else:
        gt_vis = overlay_mask(np.ones_like(rgb), gt_np, alpha=0.85)
        st.image((gt_vis * 255).astype(np.uint8), use_column_width=True)

with right:
    st.markdown("**Predicted Fire Overlay**")
    st.image((overlay * 255).astype(np.uint8), use_column_width=True)
