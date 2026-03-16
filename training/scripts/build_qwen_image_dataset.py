import argparse
import csv
import shutil
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}


def main() -> None:
    parser = argparse.ArgumentParser(description='Prepare a Qwen-Image LoRA dataset with metadata.csv and caption txt files.')
    parser.add_argument('--image-dir', required=True)
    parser.add_argument('--caption-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--copy-images', action='store_true')
    parser.add_argument('--default-caption', default='lulu the capybara')
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    caption_dir = Path(args.caption_dir)
    output_dir = Path(args.output_dir)
    train_dir = output_dir / 'train'
    train_dir.mkdir(parents=True, exist_ok=True)

    rows = [['file_name', 'text']]
    copied = 0
    for image_path in sorted(image_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        caption_path = caption_dir / f'{image_path.stem}.txt'
        caption = args.default_caption
        if caption_path.exists():
            caption = caption_path.read_text(encoding='utf-8-sig').strip() or args.default_caption

        target_image = train_dir / image_path.name
        target_caption = train_dir / f'{image_path.stem}.txt'

        if args.copy_images:
            shutil.copy2(image_path, target_image)
        else:
            if target_image.exists():
                target_image.unlink()
            target_image.symlink_to(image_path.resolve())
        target_caption.write_text(caption + '\n', encoding='utf-8')
        rows.append([image_path.name, caption])
        copied += 1

    metadata_path = train_dir / 'metadata.csv'
    with metadata_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    print(f'prepared {copied} image-caption pairs at {train_dir}')
    print(f'metadata: {metadata_path}')


if __name__ == '__main__':
    main()
