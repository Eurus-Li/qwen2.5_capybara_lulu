import argparse
import re
from pathlib import Path
from urllib.parse import unquote


def normalize_url(url: str) -> str:
    url = url.strip().strip(')>,\"\'')
    url = url.replace('&amp;', '&')
    return unquote(url)


def looks_like_image(url: str, include_hosts: list[str], include_patterns: list[str]) -> bool:
    lower = url.lower()
    if include_hosts and not any(host in lower for host in include_hosts):
        return False
    if include_patterns and not any(pattern in lower for pattern in include_patterns):
        return False
    return any(token in lower for token in ('.jpg', '.jpeg', '.png', '.webp', '.gif', '_b.', '_l.', '_qhd.'))


def main() -> None:
    parser = argparse.ArgumentParser(description='Extract image URLs from a saved local HTML file.')
    parser.add_argument('--input', required=True, help='Path to the saved HTML file.')
    parser.add_argument('--output', required=True, help='Path to the output URL list txt file.')
    parser.add_argument('--include-host', action='append', default=[], help='Only keep URLs containing this host fragment.')
    parser.add_argument('--include-pattern', action='append', default=[], help='Only keep URLs containing this pattern.')
    args = parser.parse_args()

    html_path = Path(args.input)
    output_path = Path(args.output)
    text = html_path.read_text(encoding='utf-8', errors='ignore')

    candidates = re.findall(r'https://[^\s"<>]+', text)
    urls = []
    seen = set()
    for candidate in candidates:
        url = normalize_url(candidate)
        if not looks_like_image(url, args.include_host, args.include_pattern):
            continue
        if url not in seen:
            seen.add(url)
            urls.append(url)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text('\n'.join(urls) + ('\n' if urls else ''), encoding='utf-8')
    print(f'extracted {len(urls)} image urls to {output_path}')


if __name__ == '__main__':
    main()
