"""
ComfyUI custom node: SAM3 Adapter segmentation.

Input : IMAGE  (ComfyUI standard: [B, H, W, C] float32, 0–1)
Output: MASK   (ComfyUI standard: [B, H, W]    float32, 0–1)

The node runs SAM3-Adapter inference and returns a probability map.
Set threshold > 0 to binarise; leave at 0 to get the raw probability.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F
import yaml

# ── path setup ───────────────────────────────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
_SAM_DIR = os.path.join(os.path.dirname(_HERE), "SAM-Adapter-PyTorch")
if _SAM_DIR not in sys.path:
    sys.path.insert(0, _SAM_DIR)

_DEFAULT_CONFIG = os.path.join(_SAM_DIR, "configs", "cod-sam-vit-l.yaml")
_DEFAULT_CKPT   = os.path.join(_SAM_DIR, "pretrained", "sam_vit_l_converted.pth")

# ── model cache (config_path + ckpt_path → model) ────────────────────────────
_cache: dict[tuple, tuple] = {}


def _load_model(config_path: str, checkpoint_path: str, device: str):
    key = (config_path, checkpoint_path, device)
    if key in _cache:
        return _cache[key]

    import models  # from SAM-Adapter-PyTorch

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = models.make(config["model"]).to(device)
    model.eval()

    print(f"[SAM3Adapter] Loading checkpoint: {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        state_dict = raw.get("model", raw.get("state_dict", raw))
    else:
        state_dict = raw

    # strip DDP "module." prefix
    state_dict = {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[SAM3Adapter] Missing keys (random init): {len(missing)}, "
          f"Unexpected keys (ignored): {len(unexpected)}")

    _cache[key] = (model, config)
    return model, config


# ── node class ────────────────────────────────────────────────────────────────
class SAM3AdapterNode:
    """SAM3-Adapter segmentation node for ComfyUI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "config_path": (
                    "STRING",
                    {"default": _DEFAULT_CONFIG, "multiline": False},
                ),
                "checkpoint_path": (
                    "STRING",
                    {"default": _DEFAULT_CKPT, "multiline": False},
                ),
                "threshold": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                     "tooltip": "Set > 0 to binarise the mask. 0 = raw probability."},
                ),
            }
        }

    RETURN_TYPES  = ("MASK",)
    RETURN_NAMES  = ("mask",)
    FUNCTION      = "segment"
    CATEGORY      = "segmentation"
    DESCRIPTION   = (
        "SAM3-Adapter: specialised segmentation for camouflaged objects, "
        "shadows, and polyps. Returns a probability mask (or binarised if "
        "threshold > 0)."
    )

    def segment(self, image, config_path, checkpoint_path, threshold):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, config = _load_model(config_path, checkpoint_path, device)

        inp_size = config["model"]["args"]["inp_size"]  # e.g. 1024

        # ── preprocess ───────────────────────────────────────────────────────
        # ComfyUI IMAGE: [B, H, W, C] float32 in [0, 1]
        x = image.permute(0, 3, 1, 2).to(device)  # → [B, C, H, W]

        data_norm = config.get("data_norm", {})
        inp_norm  = data_norm.get("inp", {"sub": [0.5], "div": [0.5]})
        sub = torch.tensor(inp_norm["sub"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
        div = torch.tensor(inp_norm["div"], dtype=torch.float32, device=device).view(1, -1, 1, 1)
        if sub.shape[1] == 1:
            sub = sub.expand(1, 3, 1, 1)
            div = div.expand(1, 3, 1, 1)

        x = (x - sub) / div
        x = F.interpolate(x, size=(inp_size, inp_size),
                          mode="bilinear", align_corners=False)

        # ── inference ────────────────────────────────────────────────────────
        with torch.no_grad():
            logits = model.infer(x)           # [B, 1, H, W]
            prob   = torch.sigmoid(logits)    # [B, 1, H, W]

        # ── postprocess ──────────────────────────────────────────────────────
        orig_h, orig_w = image.shape[1], image.shape[2]
        if prob.shape[2] != orig_h or prob.shape[3] != orig_w:
            prob = F.interpolate(prob, size=(orig_h, orig_w),
                                 mode="bilinear", align_corners=False)

        mask = prob[:, 0, :, :]  # [B, H, W]

        if threshold > 0.0:
            mask = (mask >= threshold).float()

        return (mask,)
