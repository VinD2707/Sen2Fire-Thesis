# app7.py
# Sen2Fire — Pixel-wise Segmentation (Tabs)
# Tab 1: Single Model Inference (clean + Train/Val/Test table only)
# Tab 2: Model Comparison (visual only) -> 5 images (RGB + GT + 3 model overlays)
#
# Pipeline: logits -> sigmoid -> threshold (validation-derived) -> visualization/product
# Patch-level logic removed.

import io
import time
import inspect
import json
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
import torch
import streamlit as st
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import matplotlib.cm as cm

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

    apply_scale_clip = bool(d.get("apply_scale_clip", False))
    scale_div = d.get("scale_div", None)
    clip = d.get("clip", None)

    return mean_13, std_13, apply_scale_clip, scale_div, clip


# ============================================================
# PATHS + MODELS (non-pretrained removed)
# ============================================================
APP_DIR = Path(__file__).resolve().parent

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

T_BEST = {
    "U-Net (JP retrained)": 0.05,
    "U-Net Eff (EfficientNet-B4)": 0.05,
    "DeepLabV3+ (EfficientNet-B4) — pretrained": 0.05,
}

PREPROCESS = {
    "U-Net (JP retrained)": {"norm_path": APP_DIR / "thesis_unet" / "norm_stats.json"},
    "U-Net Eff (EfficientNet-B4)": {"norm_path": APP_DIR / "unet_eff" / "norm_stats.json"},
    "DeepLabV3+ (EfficientNet-B4) — pretrained": {"norm_path": APP_DIR / "thesis_deeplab" / "norm_stats.json"},
}

ARCH = {
    "U-Net (JP retrained)": ("unet", "resnet18"),
    "U-Net Eff (EfficientNet-B4)": ("unet", "efficientnet-b4"),
    "DeepLabV3+ (EfficientNet-B4) — pretrained": ("deeplab", "efficientnet-b4"),
}

TEST_METRICS = {
    "U-Net (JP retrained)": {
        "test_loss": 0.0741,
        "test_precision": 0.7120,
        "test_recall": 0.6341,
        "test_f1": 0.6708,
        "test_acc": 0.9849,
        "test_pr_auc": 0.9381,
    },
    "U-Net Eff (EfficientNet-B4)": {
        "test_loss": 0.0501,
        "test_precision": 0.7206,
        "test_recall": 0.7792,
        "test_f1": 0.7488,
        "test_acc": 0.9873,
        "test_pr_auc": 0.7971,
    },
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

DISPLAY_NAME = {
    "U-Net (JP retrained)": "U-Net + Resnet18 (thr: 0.05)",
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


def overlay_probabilities(rgb01: np.ndarray, probs01: np.ndarray, alpha=0.55) -> np.ndarray:
    """
    Raw sigmoid overlay:
    intensity of red is proportional to probability (0..1).
    """
    rgb = rgb01.copy().astype(np.float32)
    p = np.clip(probs01.astype(np.float32), 0.0, 1.0)[..., None]  # (H,W,1)

    red = np.zeros_like(rgb)
    red[..., 0] = 1.0

    # stronger probs -> more red
    return np.clip((1 - alpha * p) * rgb + (alpha * p) * red, 0, 1)


def red_mask_on_white(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 > 0.5).astype(np.float32)
    out = np.ones((m.shape[0], m.shape[1], 3), dtype=np.float32)
    out[..., 1] = 1.0 - 0.85 * m
    out[..., 2] = 1.0 - 0.85 * m
    return out



def binary_mask_red_on_white(mask01: np.ndarray) -> np.ndarray:
    """
    Fire = red, Non-fire = white
    SAME for Ground Truth and Prediction
    """
    m = (mask01 > 0.5).astype(np.float32)
    out = np.ones((m.shape[0], m.shape[1], 3), dtype=np.float32)
    out[..., 1] = 1.0 - 0.85 * m   # reduce G
    out[..., 2] = 1.0 - 0.85 * m   # reduce B
    return out

def swir_for_display(x13: np.ndarray, swir_idx=11) -> np.ndarray:
    swir = x13[swir_idx]
    lo, hi = np.percentile(swir, 2), np.percentile(swir, 98)
    swir_n = np.clip((swir - lo) / (hi - lo + 1e-6), 0, 1)
    return swir_n

def aggregate_binary_mask_area(
    mask01: np.ndarray,
    block: int = 56,
    min_fire_pixels: int = 314,  # 🔥 threshold area
) -> np.ndarray:
    """
    Spatial aggregation based on MINIMUM FIRE PIXELS per block.
    """
    x = torch.from_numpy(mask01).unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)

    # sum pooling → jumlah pixel api
    pooled_sum = F.avg_pool2d(
        x,
        kernel_size=block,
        stride=block
    ) * (block * block)

    # decision based on area
    agg = (pooled_sum >= min_fire_pixels).float()

    # upsample back to original size
    up = F.interpolate(agg, size=mask01.shape, mode="nearest")

    return up[0, 0].numpy()






VIS_BLOK_CFG = {
    "U-Net (JP retrained)": {"t_best": 0.05, "min_pixel": 1000.0, "blok_pixel": 64, "swir_idx": 10},
    "U-Net Eff (EfficientNet-B4)": {"t_best": 0.05, "min_pixel": 448.0,  "blok_pixel": 56, "swir_idx": 10},
    "DeepLabV3+ (EfficientNet-B4) — pretrained": {"t_best": 0.05, "min_pixel": 1000.0, "blok_pixel": 96, "swir_idx": 10},
}

def swir_minmax_for_display(x13: np.ndarray, swir_idx: int = 10) -> np.ndarray:
    swir = x13[swir_idx].astype(np.float32)
    return stretch01(swir, p1=1, p2=99)

def reds_cmap_img(arr01: np.ndarray) -> np.ndarray:
    """
    Match notebook: imshow(arr, cmap="Reds")
    Input: 2D array (0..1)
    Output: uint8 RGB (H,W,3)
    """
    a = np.clip(arr01.astype(np.float32), 0.0, 1.0)
    rgba = cm.get_cmap("Reds")(a)     # (H,W,4)
    rgb = rgba[..., :3]               # (H,W,3)
    return (rgb * 255).astype(np.uint8)

def blockify_and_threshold_like_nb(
    pred_map: np.ndarray,
    block_size: int,
    threshold: float,
    binarize: bool = True,
    bin_thr: float = 0.5
) -> np.ndarray:
    """
    Blockify berdasarkan JUMLAH PIXEL API per block (bukan sum probabilitas).

    pred_map:
      - idealnya binary mask {0,1}
      - kalau masih probabilitas, set binarize=True (default), maka jadi (pred_map >= bin_thr)

    threshold:
      - min_pixel, yaitu minimal jumlah pixel api dalam satu block
    """
    H, W = pred_map.shape

    if binarize:
        pm = (pred_map >= float(bin_thr)).astype(np.uint8)
    else:
        pm = pred_map.astype(np.uint8)

    blocked = np.zeros((H, W), dtype=np.float32)

    thr = int(threshold)

    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            block = pm[i:i+block_size, j:j+block_size]
            fire_pixels = int(block.sum())
            if fire_pixels >= thr:
                blocked[i:i+block_size, j:j+block_size] = 1.0

    return blocked



# ============================================================
# AGGREGATION PARAMETERS (AREA-BASED)
# ============================================================
AGG_BLOCK = 56                 # 56 × 56 pixel
MIN_FIRE_PIXELS = 314          # ~10% area of 56×56 block


from scipy.ndimage import label as cc_label
from scipy.ndimage import binary_opening, binary_closing, binary_fill_holes

def stretch01(img, p1=2, p2=98):
    a = img.astype(np.float32)
    lo = np.nanpercentile(a, p1)
    hi = np.nanpercentile(a, p2)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def cca_process(mask01: np.ndarray, min_pixels=2582, connectivity=2):
    if connectivity == 1:
        structure = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
    else:
        structure = np.ones((3,3), dtype=np.uint8)

    labeled, n = cc_label(mask01.astype(bool), structure=structure)
    out = np.zeros_like(mask01, dtype=np.uint8)

    for i in range(1, n+1):
        comp = (labeled == i)
        if comp.sum() >= min_pixels:
            out[comp] = 1
    return out


BAND = {
    "GREEN": 2,
    "NIR": 7,
    "SWIR2": 11,
}

def make_guidance_rgb_from_xraw(x_raw_chw):
    g   = x_raw_chw[BAND["GREEN"]].astype(np.float32)
    nir = x_raw_chw[BAND["NIR"]].astype(np.float32)
    sw  = x_raw_chw[BAND["SWIR2"]].astype(np.float32)

    rgb = np.stack([nir, sw, g], axis=-1)  # (H,W,3)

    vmin = np.nanpercentile(rgb, 1)
    vmax = np.nanpercentile(rgb, 99)

    rgb = np.clip((rgb - vmin) / (vmax - vmin + 1e-6), 0, 1)
    return np.ascontiguousarray((rgb * 255).astype(np.uint8))


def crf_refine_probs(probs2d: np.ndarray, guide_u8: np.ndarray, crf_iters: int = 5) -> np.ndarray:
    probs2d = np.ascontiguousarray(np.clip(probs2d.astype(np.float32), 1e-6, 1 - 1e-6))
    guide_u8 = np.ascontiguousarray(guide_u8.astype(np.uint8))
    H, W = probs2d.shape

    try:
        import pydensecrf.densecrf as dcrf
        from pydensecrf.utils import unary_from_softmax

        d = dcrf.DenseCRF2D(W, H, 2)

        soft = np.zeros((2, H, W), dtype=np.float32)
        soft[1] = probs2d
        soft[0] = 1.0 - probs2d

        U = unary_from_softmax(soft)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=3, compat=3)
        d.addPairwiseBilateral(sxy=50, srgb=10, rgbim=guide_u8, compat=5)  # <- compat=5 sesuai notebook

        Q = np.array(d.inference(crf_iters), dtype=np.float32).reshape(2, H, W)
        return Q[1].astype(np.float32)

    except Exception:
        # fallback bilateral filter (butuh opencv-python)
        try:
            import cv2
            p8 = (probs2d * 255).astype(np.uint8)
            p8 = cv2.bilateralFilter(p8, d=7, sigmaColor=25, sigmaSpace=25)
            return (p8.astype(np.float32) / 255.0).astype(np.float32)
        except Exception:
            return probs2d




def infer_with_internals(model_key: str, x13: np.ndarray, t_used: float):

    device = get_device()
    model = load_model_cached(model_key)

    pp = PREPROCESS[model_key]
    mean_13, std_13, apply_scale_clip, scale_div, clip = load_norm_stats(pp["norm_path"])

    x13p = x13.astype(np.float32)
    x_raw = torch.from_numpy(x13p).unsqueeze(0).to(device)
    x_norm = normalize_batch(x_raw, mean_13, std_13)

    with torch.no_grad():
        logits = model(x_norm)
        probs  = torch.sigmoid(logits)

    # RAW sigmoid (untuk v3)
    probs_raw = probs[0, 0].detach().cpu().numpy().astype(np.float32)

    # === CRF (persis notebook) ===
    guide = make_guidance_rgb_from_xraw(x13)                 # uint8
    probs_crf = crf_refine_probs(probs_raw, guide, crf_iters=5)

    # mask_thr = (probs_crf >= float(t_used)).astype(np.uint8)
    mask_thr = (probs_raw >= float(t_used)).astype(np.uint8)

    # CCA
    mask_final = cca_process(mask_thr, min_pixels=2582)

    rgb = to_rgb_for_display(x13)

    visuals = {
        "rgb": rgb,
        "pred_prob_overlay": overlay_probabilities(rgb, probs_raw),  # v3 tebal-tipis
        "pred_bin_overlay": overlay_binary(rgb, mask_final),
    }

    internals = {
        "t_used": float(t_used),
        "mean_raw": float(probs_raw.mean()),
        "mean_crf": float(probs_crf.mean()),
    }

    return visuals, internals, probs_raw, probs_crf, mask_final


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


def pick_pr_auc_col(df: pd.DataFrame, prefix: str) -> Optional[str]:
    candidates = [
        f"{prefix}pr_auc",
        f"{prefix}prAUC",
        f"{prefix}prauc",
        f"{prefix}PRauc",
        f"{prefix}PR_AUC",
        f"{prefix}auc",
        f"{prefix}AUC",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if c.lower().startswith(prefix.lower()) and "pr" in c.lower() and "auc" in c.lower():
            return c
    return None


def best_overall_row(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty or "va_f1" not in df.columns:
        return None
    return df.loc[df["va_f1"].idxmax()]


def get_metric_value(row: pd.Series, key_candidates: List[str]) -> Any:
    for k in key_candidates:
        if k in row.index:
            return row[k]
    return np.nan


def build_train_val_test_table(df: pd.DataFrame, model_key: str) -> pd.DataFrame:
    row = best_overall_row(df)
    tm = TEST_METRICS.get(model_key, {})

    if row is None:
        return pd.DataFrame([{
            "split": "test_set",
            "loss": tm.get("test_loss", np.nan),
            "precision": tm.get("test_precision", np.nan),
            "recall": tm.get("test_recall", np.nan),
            "f1": tm.get("test_f1", np.nan),
            "acc": tm.get("test_acc", np.nan),
            "pr_auc": tm.get("test_pr_auc", np.nan),
        }])

    tr_pr_auc_col = pick_pr_auc_col(df, "tr_")
    va_pr_auc_col = pick_pr_auc_col(df, "va_")

    train = {
        "split": "train_set",
        "loss": get_metric_value(row, ["tr_loss", "train_loss"]),
        "precision": get_metric_value(row, ["tr_precision", "train_precision"]),
        "recall": get_metric_value(row, ["tr_recall", "train_recall"]),
        "f1": get_metric_value(row, ["tr_f1", "train_f1"]),
        "acc": get_metric_value(row, ["tr_acc", "train_acc", "tr_accuracy", "train_accuracy"]),
        "pr_auc": row[tr_pr_auc_col] if (tr_pr_auc_col and tr_pr_auc_col in row.index) else np.nan,
    }

    val = {
        "split": "val_set",
        "loss": get_metric_value(row, ["va_loss", "val_loss"]),
        "precision": get_metric_value(row, ["va_precision", "val_precision"]),
        "recall": get_metric_value(row, ["va_recall", "val_recall"]),
        "f1": get_metric_value(row, ["va_f1", "val_f1"]),
        "acc": get_metric_value(row, ["va_acc", "val_acc", "va_accuracy", "val_accuracy"]),
        "pr_auc": row[va_pr_auc_col] if (va_pr_auc_col and va_pr_auc_col in row.index) else np.nan,
    }

    test = {
        "split": "test_set",
        "loss": tm.get("test_loss", np.nan),
        "precision": tm.get("test_precision", np.nan),
        "recall": tm.get("test_recall", np.nan),
        "f1": tm.get("test_f1", np.nan),
        "acc": tm.get("test_acc", np.nan),
        "pr_auc": tm.get("test_pr_auc", np.nan),
    }

    return pd.DataFrame([train, val, test])


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
st.markdown("Segmentation (logits) → sigmoid → CRF → static threshold (0.05) → CCA (min_pixels=2582)")

# validate paths early
missing = []
for k, p in WEIGHTS.items():
    if not p.exists():
        missing.append(f"{k} weights missing: {p}")
for k, d in PREPROCESS.items():
    if not d["norm_path"].exists():
        missing.append(f"{k} norm_stats.json missing: {d['norm_path']}")

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

    t_used = float(T_BEST[model_key])
    # visuals, internals, probs_np, mask_np = infer_with_internals(model_key, x13, t_used=t_used)
    visuals, internals, probs_raw, probs_np, mask_np = infer_with_internals(model_key, x13, t_used=t_used)

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown("## Model Info")
    st.markdown(f"""
**Model Name:**  
{DISPLAY_NAME.get(model_key, model_key)}

**Threshold used (t_used):**  
{t_used}

**Patch Size:**  
{H} × {W}{" ✅" if size_ok else " ⚠️"}

**Input:**  
Model input: concatenation → **(13, H, W)** (12-band image + 1 aerosol)

**Output:**  
Logits → sigmoid probabilities (0–1) → threshold (**t_used**) → binary mask
""")

    st.markdown("<hr/>", unsafe_allow_html=True)
    if gt01 is not None:
        st.markdown("## Selected Patch Prediction Overview")
        st.markdown('<div class="small">metrics: loss, precision, recall, f1, accuracy, PR_AUC</div>', unsafe_allow_html=True)

        pm = compute_patch_metrics(gt01, probs_raw, mask_np)
        # pm = compute_patch_metrics(gt01, probs_np, mask_np)
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

    st.markdown("<hr/>", unsafe_allow_html=True)
    
    
    st.markdown("## Visualization")
    st.caption(
        "**Note:** Red area represents fire pixels, while white area represents non-fire pixels "
        "in both the Ground Truth and the Binary Fire Map."
    )

    # --- notebook-style params (vis-blok-all_final.ipynb) ---
    cfg = VIS_BLOK_CFG.get(model_key, {"t_best": 0.05, "min_pixel": 448.0, "blok_pixel": 56, "swir_idx": 10})
    t_vis = float(cfg["t_best"])
    min_pixel = float(cfg["min_pixel"])
    blok_pixel = int(cfg["blok_pixel"])
    swir_idx = int(cfg["swir_idx"])

    # area seperti notebook: mask = (probs >= t_best) lalu mean()
    # area_val = float((probs_np >= t_vis).astype(np.float32).mean())
    area_val = float((probs_raw >= t_vis).astype(np.float32).mean())
    patch_name = uploaded.name if hasattr(uploaded, "name") else "patch"

    v1, v2, v3, v4, v5 = st.columns(5, gap="medium")

    # SWIR
    swir_vis = swir_minmax_for_display(x13, swir_idx=swir_idx)
    v1.image(
       (swir_vis * 255).astype(np.uint8),
       caption=f"{patch_name}\nSWIR | Area={area_val:.4f}",
       use_column_width=True
    )

    # Ground Truth
    if gt01 is None:
       v2.warning("Ground Truth not found") 
    else:
        v2.image(
            reds_cmap_img(gt01),
            caption="Ground Truth",
            use_column_width=True
        )

    # Sigmoid Prediction (RAW)
    v3.image(
        reds_cmap_img(probs_raw),
        caption="Sigmoid Prediction (Raw Probability)",
        use_column_width=True
    )

    # Final Binary (CRF + Threshold + CCA)
    v4.image(
        reds_cmap_img(mask_np),
        caption="Final Binary (CRF + Static Thr + CCA)",
        use_column_width=True
    )

    # Blockified (Area-level)
    # mask_block = blockify_and_threshold_like_nb(probs_np, blok_pixel, min_pixel)
    # mask_block = blockify_and_threshold_like_nb(probs_raw, blok_pixel, min_pixel)

    mask_block = blockify_and_threshold_like_nb(
        probs_raw,
        # probs_np,
        blok_pixel,
        min_pixel,
        binarize=True,
        bin_thr=0.05
    )
    
    v5.image(
        reds_cmap_img(mask_block),
        caption=f"Blockified (k={blok_pixel})",
        use_column_width=True
    )


    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(f"## {DISPLAY_NAME.get(model_key, model_key)}")

    dfm = load_metrics_df(model_key)
    table = build_train_val_test_table(dfm, model_key)

    row = best_overall_row(dfm)
    if row is not None:
        run_str = str(row.get("run", ""))
        ep_str = str(int(row.get("epoch", -1))) if "epoch" in row.index else ""
        st.markdown(
            f'<div class="small">Train/Val taken from best epoch (best va_f1) → run: <b>{run_str}</b>, epoch: <b>{ep_str}</b>. Test is hard-coded (TEST_METRICS).</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="small">Train/Val table not found in metrics file (or `va_f1` missing). Test is still shown from TEST_METRICS.</div>',
            unsafe_allow_html=True
        )

    st.dataframe(table, use_container_width=True)

    st.markdown(
        '<div class="small" style="margin-top:10px;">Notes: This app is pixel-wise only. Patch-level aggregation is removed.</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# TAB 2 — COMPARISON (visual only)
# SWIR + GT + 3 aggregated binary predictions (56x56)
# ============================================================
with tab_compare:

    # ========================================================
    # HEADER + TOGGLE (KANAN ATAS)
    # ========================================================
    header_cols = st.columns([3, 2])

    with header_cols[0]:
        st.subheader("Model Comparison (Visual Only)")

    with header_cols[1]:
        viz_mode = st.radio(
            "Visualization Mode",
            ["Visualize by Block", "Visualize by Binary Mask"],
            horizontal=True,
            key="viz_mode_compare",
        )

        # viz_mode = st.radio(
        #     "Visualization Mode",
        #     [
        #         "Visualize by Sigmoid Binary Mask (thr=0.05)",
        #         "Visualize by Postprocessed Mask (CRF + Thr + CCA)",
        #         "Visualize by Block (from Sigmoid Binary)",
        #     ],
        #     horizontal=True,
        #     key="viz_mode_compare",
        # )

    # ========================================================
    # MODEL SELECTION
    # ========================================================
    all_models = list(WEIGHTS.keys())

    models = st.multiselect(
        "Select models to compare (max 3 will be shown)",
        all_models,
        default=all_models,
        key="compare_models",
        format_func=lambda k: DISPLAY_NAME.get(k, k),
    )

    # ========================================================
    # FILE UPLOAD
    # ========================================================
    uploaded2 = st.file_uploader(
        "Upload test patch (.npz)",
        type=["npz"],
        key="compare_upload",
    )

    if uploaded2 is None:
        st.info("Upload a .npz file to compare models.")
        st.stop()

    try:
        x13c, gt01c = load_npz_from_bytes(uploaded2.getvalue())
    except Exception as e:
        st.error(f"Failed to read NPZ: {e}")
        st.stop()

    # ========================================================
    # SWIR VISUALIZATION
    # ========================================================
    swir = swir_minmax_for_display(x13c, swir_idx=10)

    models_ordered = [m for m in all_models if m in models]
    models_show = models_ordered[:3]

    st.markdown("### Model Comparison")
    st.caption(
        "**Note:** Ground Truth, raw sigmoid, and blockified maps use `Reds` colormap"
    )

    cols = st.columns(5, gap="medium")

    # ========================================================
    # 1 SWIR
    # ========================================================
    cols[0].image(
        (swir * 255).astype(np.uint8),
        caption="SWIR",
        use_column_width=True
    )

    # ========================================================
    # 2 GROUND TRUTH
    # ========================================================
    if gt01c is None:
        cols[1].warning("Ground Truth not found (missing key: label)")
    else:
        cols[1].image(
            reds_cmap_img(gt01c),
            caption="Ground Truth",
            use_column_width=True
        )

    # ========================================================
    # 3–5 MODEL PREDICTIONS
    # ========================================================
    for j in range(3):

        col_idx = 2 + j

        if j < len(models_show):

            m = models_show[j]

            # inference
            t_model = float(T_BEST[m])
            _, _, probs_m, _, mask_final_m = infer_with_internals(
                m,
                x13c,
                t_used=t_model
            )

            # notebook config per model
            cfgm = VIS_BLOK_CFG.get(
                m,
                {
                    "t_best": 0.05,
                    "min_pixel": 448.0,
                    "blok_pixel": 56,
                    "swir_idx": 10
                }
            )

            km = int(cfgm["blok_pixel"])
            minpm = float(cfgm["min_pixel"])

            # =================================================
            # VISUALIZATION MODE SWITCH
            # =================================================
            if viz_mode == "Visualize by Block":
                pred_vis = blockify_and_threshold_like_nb(
                    probs_m,
                    km,
                    minpm,
                    binarize=True,
                    bin_thr=t_model  # <--- PENTING: 0.05, bukan 0.5
                )
                caption_txt = (
                    f"{DISPLAY_NAME.get(m, m)}\n"
                    f"Prediction (Blockified k={km})"
                )
            else:
                # ini hasil postprocessing yang sama dengan tab 1 v4 (mask_final)
                pred_vis = mask_final_m.astype(np.uint8)
                caption_txt = (
                    f"{DISPLAY_NAME.get(m, m)}\n"
                    f"Prediction (CRF + Thr + CCA)"
                )


            cols[col_idx].image(
                reds_cmap_img(pred_vis),
                caption=caption_txt,
                use_column_width=True
            )

        else:
            cols[col_idx].info(
                "Select more models (up to 3) to fill this slot."
            )

