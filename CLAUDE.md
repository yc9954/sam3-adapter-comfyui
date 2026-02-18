# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch research codebase that adapts SAM (Segment Anything Model) — specifically SAM3 — to challenging segmentation tasks:
- **Camouflaged Object Detection (COD)** — datasets: COD10K, CAMO, CHAMELEON
- **Shadow Detection** — dataset: ISTD
- **Polyp/Medical Segmentation** — dataset: Kvasir

The core idea is parameter-efficient fine-tuning: only adapter/prompt-generator weights in the image encoder are trained; the backbone is frozen.

## Setup

**1. Install dependencies (from `SAM-Adapter-PyTorch/`):**
```bash
pip install -r requirements.txt
```

**2. Download and convert the official SAM ViT-L checkpoint (~1.2 GB):**
```bash
cd SAM-Adapter-PyTorch
python convert_checkpoint.py
# Downloads pretrained/sam_vit_l_0b3195.pth → converts to pretrained/sam_vit_l_converted.pth
```

**3. Place datasets** under `SAM-Adapter-PyTorch/data/`:
- Kvasir-SEG: `data/kvasir-seg/train/{images,masks}/` and `data/kvasir-seg/test/{images,masks}/`
- COD: `data/cod/TrainDataset/{Image,GT}/` and `data/cod/TestDataset/{Image,GT}/`
- ISTD: `data/ISTD_Dataset/test/{test_A,test_B}/`

## Commands

All commands run from `SAM-Adapter-PyTorch/`.

**Training (multi-GPU, 4 GPUs):**
```bash
bash train.sh
# or manually:
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
  --nnodes 1 --nproc_per_node 4 \
  train.py --config configs/cod-sam-vit-l.yaml
```

**Inference/Evaluation:**
```bash
python test.py \
  --config configs/cod-sam-vit-l.yaml \
  --model pretrained/sam_vit_l_converted.pth \
  --save_path results/vis
```

## ComfyUI Node

The node is in `comfyui_node/` and is already symlinked to ComfyUI:
```
/Users/iyuchan/Documents/ComfyUI/custom_nodes/sam3-adapter → comfyui_node/
```

- **Node name**: `SAM3Adapter` (category: `segmentation`)
- **Input**: IMAGE (ComfyUI standard [B,H,W,C] float32)
- **Output**: MASK ([B,H,W] float32 probability, binarised if threshold > 0)
- The node uses `configs/cod-sam-vit-l.yaml` and `pretrained/sam_vit_l_converted.pth` by default
- Model is loaded lazily and cached on first use

## Architecture

### Registry Pattern
Both `models/` and `datasets/` use a decorator-based registry:
```python
@register('sam')
class SAM(nn.Module): ...
# Instantiated via: models.make(config['model'])
```

### Model (`models/sam.py`)
- Wraps `ImageEncoderViT` (ViT-L backbone) from `models/mmseg/models/sam/image_encoder.py`
- `ImageEncoderViT` includes a `PromptGenerator` adapter (the only trainable part during fine-tuning)
- `MaskDecoder` + `TwoWayTransformer` for mask prediction
- Key method: `infer(input)` — takes `[B,3,H,W]` normalised tensor, returns `[B,1,H,W]` logits
- `forward(inp, gt)` — runs inference and returns scalar loss (for training)

### Checkpoint Strategy
The public SAM ViT-L checkpoint (`sam_vit_l_0b3195.pth`) is converted by `convert_checkpoint.py`:
- `image_encoder.*` + `mask_decoder.*` → kept as-is (467 keys)
- `prompt_encoder.pe_layer.*` → remapped to `pe_layer.*`
- `prompt_encoder.no_mask_embed.weight` → remapped to `no_mask_embed.weight`
- 54 adapter keys (`image_encoder.prompt_generator.*`) are NOT in the official checkpoint → randomly initialised

### Training Pipeline (`train.py`)
- PyTorch DDP with NCCL backend (must launch via `torchrun`)
- `torch.distributed.init_process_group` runs at module import time (not inside `__main__`)
- AdamW + CosineAnnealingLR; checkpoint keys remapped from `detector.backbone.*` → `image_encoder.*`
- Only `prompt_generator` params in `image_encoder` are kept trainable

### Evaluation Metrics (`utils.py`)
| `eval_type` | Metrics |
|---|---|
| `kvasir` | Dice, IoU |
| `cod` | S-measure, E-measure, Weighted F-measure, MAE |
| `ber` | Balance Error Rate (shadow) |
| `f1` | F1, AUC |
| `fmeasure` | F-measure, MAE |

### Input Normalisation
Config `data_norm.inp`: `sub=[0.5], div=[0.5]` → maps `[0,1]` to `[-1,1]`. Input must be resized to `inp_size=1024` before passing to the model.
