#!/usr/bin/env python3
"""
Download and convert the official SAM ViT-L checkpoint to work with SAM3-Adapter.

Key remapping:
  image_encoder.*                                      → kept as-is
  mask_decoder.*                                       → kept as-is
  prompt_encoder.pe_layer.positional_encoding_*        → pe_layer.*
  prompt_encoder.no_mask_embed.weight                  → no_mask_embed.weight
  (other prompt_encoder.* keys)                        → dropped (not used)

The image_encoder.prompt_generator.* weights (adapter layers) are NOT in the
official checkpoint and will be randomly initialised at inference time.
"""

import os
import sys
import urllib.request
import torch

SAM_VIT_L_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
DEFAULT_INPUT  = "pretrained/sam_vit_l_0b3195.pth"
DEFAULT_OUTPUT = "pretrained/sam_vit_l_converted.pth"


def _reporthook(count, block_size, total_size):
    if total_size > 0:
        pct     = min(count * block_size * 100 // total_size, 100)
        mb_done = count * block_size / (1024 ** 2)
        mb_tot  = total_size / (1024 ** 2)
        print(f"\r  {pct:3d}%  {mb_done:.0f}/{mb_tot:.0f} MB", end="", flush=True)


def download(url: str, dest: str):
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    print(f"Downloading {url}\n  → {dest}")
    urllib.request.urlretrieve(url, dest, _reporthook)
    print()


def convert(src: str, dst: str):
    print(f"Converting {src} ...")
    raw = torch.load(src, map_location="cpu", weights_only=False)
    state_dict = raw.get("model", raw) if isinstance(raw, dict) else raw

    new_sd, skipped = {}, []
    for k, v in state_dict.items():
        if k.startswith("image_encoder.") or k.startswith("mask_decoder."):
            new_sd[k] = v
        elif k == "prompt_encoder.pe_layer.positional_encoding_gaussian_matrix":
            new_sd["pe_layer.positional_encoding_gaussian_matrix"] = v
        elif k == "prompt_encoder.no_mask_embed.weight":
            new_sd["no_mask_embed.weight"] = v
        else:
            skipped.append(k)

    print(f"  Kept {len(new_sd)} keys.  Skipped {len(skipped)}: {skipped}")
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    torch.save(new_sd, dst)
    print(f"  Saved → {dst}")


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    dst = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT

    if not os.path.exists(src):
        download(SAM_VIT_L_URL, src)

    convert(src, dst)
    print("\nDone. Set sam_checkpoint in your config to:", dst)
