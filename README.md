# qwen2.5_capybara_lulu

Created by Peggy, a pig.
Here is the world for Capybara Lulu.
Include web-scraping, data cleaning, and LLM training.
The purpose is to generate more Lulu so our world will be full of Lulu.

End-to-end local pipeline for Lulu capybara image generation with Qwen-Image LoRA.

This repository is structured for portability: pull it on a new machine, attach your dataset, and train.

## What is included

- Dataset collection tools (authorized-source crawler and local HTML extractors)
- Caption generation with Qwen2.5-VL
- Caption cleaning focused on Lulu identity consistency
- Qwen-Image LoRA training launchers
- Evaluation scripts and prompt suite

## What is intentionally NOT in GitHub

The following are ignored on purpose:

- Raw private dataset images (`data/`)
- Generated captions and training pairs (`training/captions`, `training/dataset`)
- Training outputs/checkpoints (`training/outputs`)
- Local cloned dependency repo (`external/`)

Reason: keep repository lightweight and reproducible while private data stays local.

## Recommended machine specs

Minimum:

- NVIDIA GPU with 8 GB VRAM (low-VRAM script)
- Python 3.10+
- CUDA-compatible PyTorch install

Preferred:

- 16 GB+ VRAM for faster, simpler training
- Linux server (better stability for long runs)

## Quick start on a new server

### 1. Clone

```bash
git clone https://github.com/Eurus-Li/qwen2.5_capybara_lulu.git
cd qwen2.5_capybara_lulu
```

### 2. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 3. Install dependencies and training backend

```bash
bash training/scripts/setup_server.sh
```

This installs:

- caption dependencies from `training/requirements-caption.txt`
- `DiffSynth-Studio` under `external/`
- editable `diffsynth` package

### 4. Put your private dataset in place

Expected image location:

```text
data/raw/
```

Supported image extensions include `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`.

If your dataset already has image-caption pairs, you can skip caption generation and adapt `training/dataset/.../metadata.csv` directly.

## Full training pipeline

### Step A: Generate captions

```bash
python training/scripts/generate_captions_qwen_vl.py \
  --input-dir data/raw \
  --output-dir training/captions \
  --trigger "lulu the capybara" \
  --language en
```

### Step B: Clean captions (identity + action/scene emphasis)

```bash
python training/scripts/clean_lulu_captions.py \
  --caption-dir training/captions \
  --trigger "lulu the capybara"
```

### Step C: Build Qwen-Image training dataset format

```bash
python training/scripts/build_qwen_image_dataset.py \
  --image-dir data/raw \
  --caption-dir training/captions \
  --output-dir training/dataset/qwen_image_lora \
  --copy-images
```

This creates:

```text
training/dataset/qwen_image_lora/train/
  *.jpg|*.png|*.gif
  *.txt
  metadata.csv
```

### Step D1: Low-VRAM training (8 GB friendly, two-stage)

```bash
bash training/scripts/train_qwen_image_lora_lowvram.sh
```

Stage split:

- `sft:data_process` (cache)
- `sft:train` (actual LoRA training)

Resume helpers:

```bash
ONLY_DATA_PROCESS=1 bash training/scripts/train_qwen_image_lora_lowvram.sh
ONLY_TRAIN=1 bash training/scripts/train_qwen_image_lora_lowvram.sh
```

### Step D2: High-VRAM training (faster, one-stage)

```bash
bash training/scripts/train_qwen_image_lora_highvram.sh
```

Tune for stronger GPUs with env vars:

```bash
LORA_RANK=128 EPOCHS=6 TRAIN_REPEAT=30 bash training/scripts/train_qwen_image_lora_highvram.sh
```

## Windows training entry points

PowerShell versions are available:

- `training/scripts/train_qwen_image_lora_lowvram.ps1`
- `training/scripts/train_qwen_image_lora.ps1`

## Evaluate trained LoRA quality

### Single prompt test

```bash
python training/scripts/validate_qwen_image_lora.py \
  --lora training/outputs/qwen_image_lora_lowvram/model/epoch-4.safetensors \
  --prompt "lulu the capybara happily bathing in a round tub" \
  --output training/eval/single_test.png \
  --low-vram
```

### Batch prompt suite

```bash
python training/scripts/run_lulu_eval_suite.py \
  --lora training/outputs/qwen_image_lora_lowvram/model/epoch-4.safetensors \
  --prompts training/configs/lulu_eval_prompts.json \
  --output-dir training/eval/run_01 \
  --low-vram
```

Evaluation checklist:

- see `training/LULU_EVAL.md`

## Common issues

1. No output for long time at start
- Usually first-time model download + initialization. Keep process running and monitor GPU/disk.

2. OOM on 8 GB GPU
- Use low-VRAM script.
- Lower `LORA_RANK` (for example `16`).
- Lower `TRAIN_REPEAT`.
- Close GUI/GPU-heavy apps.

3. Slow training
- Expected on low-VRAM offload mode.
- Prefer Linux + larger VRAM if possible.

4. `main` branch is clean but data is "missing"
- By design. Data is private/local and ignored in `.gitignore`.

## Repository map

- `dataset_tools/`: source collection scripts
- `manifests/`: URL manifests
- `training/scripts/`: caption/train/eval scripts
- `training/configs/`: accelerate and eval configs
- `training/LULU_EVAL.md`: quality criteria

## License and data responsibility

This repository only provides pipeline code. You are responsible for ensuring rights/authorization for any dataset used in training.
