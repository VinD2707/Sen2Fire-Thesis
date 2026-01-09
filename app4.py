# app4.py
# Sen2Fire — Pixel-wise Segmentation
# Two modes via tabs:
# 1) Single Model Inference
# 2) Model Comparison (visual only)
#
# Pipeline: logits -> sigmoid -> threshold (validation-derived) -> visualization
# Patch-level logic removed.

import io
import time
import inspect
from pathlib import Path

import numpy as np
import torch
import streamlit as st
import segmentation_models_pytorch as smp


# ============================================================
# MODEL REGISTRY
# ============================================================
APP_DIR = Path(__file__).resolve().parent

MODEL_REGISTRY = {
    "U-Net (JP retrained)": {
        "weights": APP_DIR / "thesis_unet" / "unet_best_retrained2.pth",
        "arch": "unet",
        "encoder": "resnet18",
        "t_best": 0.15,
    },
    "U-Net Eff (EfficientNet-B4)": {
        "weights": APP_DIR / "unet_eff" / "efficientb4_pretrained.pth",
        "arch": "unet",
        "encoder": "efficientnet-b4",
        "t_best": 0.15,
    },
    "DeepLabV3+ (EfficientNet-B4)": {
        "weights": APP_DIR / "thesis_deeplab" / "deeplab-b4.pth",
        "arch": "deeplab",
        "encoder": "efficientnet-b4",
        "t_best": 0.15,
    },
}

RGB_IDXS = (0, 1, 2)
MEAN_13 = [0.0] * 13
STD_13 = [1.0] * 13


# ============================================================
# HELPERS
# ============================================================
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalize(x):
    mean = torch.tensor(MEAN_13, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor(STD_13, device=x.device).view(1, -1, 1, 1)
    return (x - mean) / (std + 1e-6)


def build_model(cfg):
    if cfg["arch"] == "unet":
        return smp.Unet(
            encoder_name=cfg["encoder"],
            encoder_weights="imagenet",
            in_channels=13,
            classes=1,
            activation=None,
        )
    if cfg["arch"] == "deeplab":
        return smp.DeepLabV3Plus(
            encoder_name=cfg["encoder"],
            encoder_weights="imagenet",
            in_channels=13,
            classes=1,
            activation=None,
        )
    raise ValueError("Unknown architecture")


@st.cache_resource
def load_model(model_name):
    cfg = MODEL_REGISTRY[model_name]
    device = get_device()
    model = build_model(cfg).to(device)

    load_kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kwargs["weights_only"] = False

    ckpt = torch.load(cfg["weights"], **load_kwargs)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def load_npz(file):
    with np.load(io.BytesIO(file)) as data:
        img = data["image"].astype(np.float32)
        aer = data["aerosol"].astype(np.float32)
    aer = aer[None, ...]
    return np.concatenate([img, aer], axis=0)


def rgb_composite(x13):
    rgb = np.stack([x13[i] for i in RGB_IDXS], axis=-1)
    out = np.zeros_like(rgb)
    for c in range(3):
        lo, hi = np.percentile(rgb[..., c], 2), np.percentile(rgb[..., c], 98)
        out[..., c] = np.clip((rgb[..., c] - lo) / (hi - lo + 1e-6), 0, 1)
    return out


def infer(model_name, x13):
    cfg = MODEL_REGISTRY[model_name]
    device = get_device()
    model = load_model(model_name)

    x = torch.from_numpy(x13).unsqueeze(0).to(device)
    x = normalize(x)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)
        mask = (probs >= cfg["t_best"]).float()

    return (
        logits[0, 0].cpu().numpy(),
        probs[0, 0].cpu().numpy(),
        mask[0, 0].cpu().numpy(),
    )


def overlay(rgb, mask, alpha=0.35):
    red = np.zeros_like(rgb)
    red[..., 0] = 1
    return (1 - alpha * mask[..., None]) * rgb + alpha * mask[..., None] * red


# ============================================================
# PAGE
# ============================================================
st.set_page_config(layout="wide")
st.title("Sen2Fire — Pixel-wise Segmentation")

st.warning(
    "Normalization is currently PLACEBO (mean=0, std=1). "
    "This is acceptable for UI & pipeline wiring."
)

tab_single, tab_compare = st.tabs(
    ["Single Model Inference", "Model Comparison"]
)

# ============================================================
# TAB 1 — SINGLE MODEL
# ============================================================
with tab_single:
    st.subheader("Single Model Inference")

    model_name = st.selectbox(
        "Select model",
        MODEL_REGISTRY.keys(),
    )

    file = st.file_uploader(
        "Upload test patch (.npz)",
        type="npz",
        key="single_upload",
    )

    if file:
        x13 = load_npz(file.read())
        rgb = rgb_composite(x13)

        logits, probs, mask = infer(model_name, x13)
        ov = overlay(rgb, mask)

        st.markdown("### Visualization")
        c1, c2, c3, c4 = st.columns(4)
        c1.image(rgb, caption="RGB composite")
        c2.image(probs, caption="Probability")
        c3.image(mask, caption="Binary mask")
        c4.image(ov, caption="Predicted overlay")

# ============================================================
# TAB 2 — COMPARISON
# ============================================================
with tab_compare:
    st.subheader("Model Comparison (Visual)")

    models = st.multiselect(
        "Select models to compare",
        MODEL_REGISTRY.keys(),
        default=list(MODEL_REGISTRY.keys())[:2],
    )

    file = st.file_uploader(
        "Upload test patch (.npz)",
        type="npz",
        key="compare_upload",
    )

    if file and len(models) > 0:
        x13 = load_npz(file.read())
        rgb = rgb_composite(x13)

        st.markdown("### Comparison Results")

        cols = st.columns(len(models) + 1)
        cols[0].image(rgb, caption="RGB")

        for i, m in enumerate(models):
            _, _, mask = infer(m, x13)
            cols[i + 1].image(
                overlay(rgb, mask),
                caption=m,
            )
