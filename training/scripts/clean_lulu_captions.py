import argparse
import csv
import re
from pathlib import Path

LOOK_TOKENS = [
    'cute cartoon capybara character',
    'round face',
    'brown fur',
    'small black eyes',
    'tiny rounded ears',
    'short limbs',
    'soft warm brown and beige color palette',
]

STOP_PHRASES = [
    'surrounded by',
    'the text',
    'subtitle',
    'watermark',
    'caption',
    'logo',
    'framed by',
    'the scene is',
    'looking at the viewer',
    'looking at camera',
]

ACTION_PATTERNS = [
    r'\b(smiling|hugging|holding|standing|sitting|running|riding|sleeping|eating|crying|waving|lying|jumping|wearing|walking|bathing|washing|dancing|pointing|working|shopping|celebrating)\b[^.]*',
]

SCENE_PATTERNS = [
    r'\b(in|at|on|inside|outside|under|beside|near|against)\b[^.]*',
]


def split_parts(text: str) -> list[str]:
    parts = [part.strip() for part in re.split(r',|\.', text) if part.strip()]
    return parts


def sanitize_text(text: str) -> str:
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace('Lulu the capybara', 'lulu the capybara')
    text = text.replace('Lulu the Capybara', 'lulu the capybara')
    return text


def extract_action_scene(text: str) -> tuple[str, str]:
    lowered = text.lower()
    action = ''
    scene = ''
    for pattern in ACTION_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            action = match.group(0).strip(' ,.')
            break
    for pattern in SCENE_PATTERNS:
        match = re.search(pattern, lowered)
        if match:
            scene = match.group(0).strip(' ,.')
            break
    return action, scene


def collapse_part(part: str) -> str:
    part = re.sub(r'\b(in|at|on)\b +\1\b', r'\1', part)
    return re.sub(r'\s+', ' ', part).strip(' ,.')


def clean_caption(raw: str, trigger: str) -> str:
    text = sanitize_text(raw)
    lower = text.lower()

    for phrase in STOP_PHRASES:
        lower = re.sub(rf'[^,.]*{re.escape(phrase)}[^,.]*[,.]?', ' ', lower)
    lower = re.sub(r'\s+', ' ', lower).strip(' ,.')

    action, scene = extract_action_scene(lower)

    parts = [trigger]
    parts.extend(LOOK_TOKENS)
    if action:
        parts.append(action)
    if scene and scene not in action:
        parts.append(scene)

    deduped = []
    seen = set()
    for part in parts:
        part = collapse_part(part)
        if not part:
            continue
        if part and any(part != other and part in other for other in deduped):
            continue
        if part not in seen:
            seen.add(part)
            deduped.append(part)

    phrase_parts = []
    phrase_seen = set()
    for chunk in ', '.join(deduped).split(','):
        chunk = chunk.strip()
        if not chunk or chunk in phrase_seen:
            continue
        phrase_seen.add(chunk)
        phrase_parts.append(chunk)
    return ', '.join(phrase_parts)


def update_metadata(metadata_path: Path, caption_dir: Path) -> None:
    with metadata_path.open('r', encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.reader(handle))
    header = rows[0]
    updated = [header]
    for row in rows[1:]:
        if not row:
            continue
        file_name = row[0]
        caption_path = caption_dir / f'{Path(file_name).stem}.txt'
        if caption_path.exists():
            row[1] = caption_path.read_text(encoding='utf-8-sig').strip()
        updated.append(row)
    with metadata_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerows(updated)


def main() -> None:
    parser = argparse.ArgumentParser(description='Normalize Lulu captions for stronger character consistency in LoRA training.')
    parser.add_argument('--caption-dir', required=True)
    parser.add_argument('--trigger', default='lulu the capybara')
    parser.add_argument('--metadata', help='Optional metadata.csv to update after cleaning.')
    args = parser.parse_args()

    caption_dir = Path(args.caption_dir)
    for path in sorted(caption_dir.glob('*.txt')):
        raw = path.read_text(encoding='utf-8-sig')
        cleaned = clean_caption(raw, args.trigger)
        path.write_text(cleaned + '\n', encoding='utf-8')
        print(f'cleaned {path.name}')

    if args.metadata:
        update_metadata(Path(args.metadata), caption_dir)
        print(f'updated metadata: {args.metadata}')


if __name__ == '__main__':
    main()
