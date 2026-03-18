import argparse
from pathlib import Path

import torch
from diffsynth.pipelines.qwen_image import ModelConfig, QwenImagePipeline


def build_pipeline(low_vram: bool) -> QwenImagePipeline:
    if low_vram:
        vram_config = {
            'offload_dtype': 'disk',
            'offload_device': 'disk',
            'onload_dtype': torch.float8_e4m3fn,
            'onload_device': 'cpu',
            'preparing_dtype': torch.float8_e4m3fn,
            'preparing_device': 'cuda',
            'computation_dtype': torch.bfloat16,
            'computation_device': 'cuda',
        }
        model_configs = [
            ModelConfig(model_id='Qwen/Qwen-Image', origin_file_pattern='transformer/diffusion_pytorch_model*.safetensors', **vram_config),
            ModelConfig(model_id='Qwen/Qwen-Image', origin_file_pattern='text_encoder/model*.safetensors', **vram_config),
            ModelConfig(model_id='Qwen/Qwen-Image', origin_file_pattern='vae/diffusion_pytorch_model.safetensors', **vram_config),
        ]
        return QwenImagePipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device='cuda',
            model_configs=model_configs,
            tokenizer_config=ModelConfig(model_id='Qwen/Qwen-Image', origin_file_pattern='tokenizer/'),
            vram_limit=torch.cuda.mem_get_info('cuda')[1] / (1024 ** 3) - 0.5,
        )

    return QwenImagePipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device='cuda',
        model_configs=[
            ModelConfig(model_id='Qwen/Qwen-Image', origin_file_pattern='transformer/diffusion_pytorch_model*.safetensors'),
            ModelConfig(model_id='Qwen/Qwen-Image', origin_file_pattern='text_encoder/model*.safetensors'),
            ModelConfig(model_id='Qwen/Qwen-Image', origin_file_pattern='vae/diffusion_pytorch_model.safetensors'),
        ],
        tokenizer_config=ModelConfig(model_id='Qwen/Qwen-Image', origin_file_pattern='tokenizer/'),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate one validation image with a trained Qwen-Image LoRA.')
    parser.add_argument('--lora', required=True, help='Path to the trained LoRA checkpoint, e.g. epoch-4.safetensors')
    parser.add_argument('--prompt', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--negative-prompt', default='')
    parser.add_argument('--low-vram', action='store_true')
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(args.low_vram)
    pipe.load_lora(pipe.dit, args.lora)
    image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        num_inference_steps=args.steps,
    )
    image.save(output_path)
    print(f'saved {output_path}')


if __name__ == '__main__':
    main()
