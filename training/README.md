Qwen-Image low-VRAM training notes

Prepared assets
- Dataset: .\training\dataset\qwen_image_lora\train
- Captions: .\training\captions
- Accelerate config: .\training\configs\accelerate_single_gpu.yaml
- Launch script: .\training\scripts\train_qwen_image_lora_lowvram.ps1

Recommended command
powershell -ExecutionPolicy Bypass -File .\training\scripts\train_qwen_image_lora_lowvram.ps1

Useful switches
- Only preprocess/cache stage:
  powershell -ExecutionPolicy Bypass -File .\training\scripts\train_qwen_image_lora_lowvram.ps1 -OnlyDataProcess
- Resume training stage after cache exists:
  powershell -ExecutionPolicy Bypass -File .\training\scripts\train_qwen_image_lora_lowvram.ps1 -OnlyTrain

Current low-VRAM defaults
- max_pixels: 1048576
- lora_rank: 32
- epochs: 4
- train dataset_repeat: 20
- dataset workers: 2
- gradient checkpointing offload: enabled

Tips for 8GB VRAM
- Close browsers, games, and GPU-heavy apps before training.
- If you still OOM, lower LoraRank to 16 and TrainRepeat to 10.
- Expect the first run to download large Qwen-Image weights from Hugging Face.
