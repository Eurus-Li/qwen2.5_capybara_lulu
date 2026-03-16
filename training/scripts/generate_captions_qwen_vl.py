import argparse
import json
import sys
import re
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}


def iter_images(input_dir: Path) -> Iterable[Path]:
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def normalize_caption(text: str) -> str:
    text = text.strip()
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^["\']|["\']$', '', text)
    return text


def build_prompt(trigger: str, language: str) -> str:
    if language == 'zh':
        return (
            '请用一句适合文生图训练的中文 caption 描述这张图。'
            '必须准确描述主体、表情、动作、构图和明显场景。'
            f'主体统一写成“{trigger}”。'
            '不要编造看不见的细节，不要加解释，不要分点。'
        )
    return (
        'Write one concise training caption for this image for a text-to-image LoRA. '
        'Describe the main subject, expression, pose or action, framing, and obvious setting. '
        f'Always name the subject as "{trigger}". '
        'Ignore any overlaid text, subtitles, or watermarks. Do not invent unseen details. Output caption only.'
    )


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    parser = argparse.ArgumentParser(description='Generate captions with a local Qwen2.5-VL model.')
    parser.add_argument('--input-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-3B-Instruct')
    parser.add_argument('--trigger', default='lulu the capybara')
    parser.add_argument('--language', choices=['zh', 'en'], default='en')
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], default='auto')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = 'auto' if args.device == 'auto' else None
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained(args.model)

    prompt_text = build_prompt(args.trigger, args.language)

    image_paths = list(iter_images(input_dir))
    if args.limit:
        image_paths = image_paths[: args.limit]

    for index, image_path in enumerate(image_paths, start=1):
        caption_path = output_dir / f'{image_path.stem}.txt'
        if caption_path.exists() and not args.overwrite:
            print(f'[{index}/{len(image_paths)}] skip {image_path.name}')
            continue

        image = Image.open(image_path).convert('RGB')
        messages = [
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': prompt_text},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors='pt')
        if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()):
            inputs = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

        with torch.inference_mode():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        generated_trimmed = [out[len(inp):] for inp, out in zip(inputs['input_ids'], generated)]
        caption = processor.batch_decode(generated_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        caption = normalize_caption(caption)
        if args.trigger not in caption:
            if args.language == 'zh':
                caption = f'{args.trigger}，{caption}'
            else:
                caption = f'{args.trigger}, {caption}'
        caption_path.write_text(caption + '\n', encoding='utf-8')
        print(f'[{index}/{len(image_paths)}] wrote {caption_path.name}')

    manifest = {
        'model': args.model,
        'trigger': args.trigger,
        'language': args.language,
        'count': len(image_paths),
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
    }
    (output_dir / '_caption_run.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
