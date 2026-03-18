import argparse
import json
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
    parser = argparse.ArgumentParser(description='Run a batch of Lulu LoRA validation prompts.')
    parser.add_argument('--lora', required=True)
    parser.add_argument('--prompts', required=True, help='JSON file with prompt objects')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--low-vram', action='store_true')
    args = parser.parse_args()

    prompts = json.loads(Path(args.prompts).read_text(encoding='utf-8-sig'))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pipe = build_pipeline(args.low_vram)
    pipe.load_lora(pipe.dit, args.lora)

    manifest = []
    for item in prompts:
        slug = item['id']
        prompt = item['prompt']
        seed = item.get('seed', 0)
        negative_prompt = item.get('negative_prompt', '')
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            num_inference_steps=args.steps,
        )
        output_path = output_dir / f'{slug}.png'
        image.save(output_path)
        manifest.append({
            'id': slug,
            'prompt': prompt,
            'seed': seed,
            'output': str(output_path),
            'check': item.get('check', []),
        })
        print(f'saved {output_path.name}')

    (output_dir / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
