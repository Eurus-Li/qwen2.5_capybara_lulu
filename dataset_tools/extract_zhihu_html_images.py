import argparse
import re
from pathlib import Path
from urllib.parse import unquote

IMAGE_HOSTS = (
    'pic1.zhimg.com',
    'pic2.zhimg.com',
    'pic3.zhimg.com',
    'pic4.zhimg.com',
    'picx.zhimg.com',
    'pica.zhimg.com',
)


def normalize_url(url: str) -> str:
    url = url.strip().strip(')>,\"\'')
    url = url.replace('&amp;', '&')
    url = unquote(url)
    return url


def looks_like_image(url: str) -> bool:
    lower = url.lower()
    return any(host in lower for host in IMAGE_HOSTS) and any(
        token in lower for token in ('.jpg', '.jpeg', '.png', '.webp', '.gif', '_b.', '_l.')
    )


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract Zhihu image URLs from a saved local HTML file.')
    parser.add_argument('--input', required=True, help='Path to the saved HTML file.')
    parser.add_argument('--output', required=True, help='Path to the output URL list txt file.')
    args = parser.parse_args()

    html_path = Path(args.input)
    output_path = Path(args.output)
    text = html_path.read_text(encoding='utf-8', errors='ignore')

    candidates = re.findall(r'https://[^\s"<>]+', text)
    urls = []
    seen = set()
    for candidate in candidates:
        url = normalize_url(candidate)
        if not looks_like_image(url):
            continue
        if url not in seen:
            seen.add(url)
            urls.append(url)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(urls) + ('\n' if urls else ''), encoding='utf-8')
    print(f'extracted {len(urls)} image urls to {output_path}')


if __name__ == '__main__':
    main()
