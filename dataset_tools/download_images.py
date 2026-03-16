import argparse
import csv
import hashlib
import imghdr
import json
import re
import time
from html import unescape
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0 Safari/537.36"
)


IMG_TAG_RE = re.compile(r'''<img[^>]+src=["']([^"']+)["']''', re.IGNORECASE)
ANCHOR_TAG_RE = re.compile(r'''<a[^>]+href=["']([^"']+)["']''', re.IGNORECASE)
OG_IMAGE_RE = re.compile(
    r'''<meta[^>]+property=["']og:image["'][^>]+content=["']([^"']+)["']''',
    re.IGNORECASE,
)
TWITTER_IMAGE_RE = re.compile(
    r'''<meta[^>]+name=["']twitter:image["'][^>]+content=["']([^"']+)["']''',
    re.IGNORECASE,
)


def fetch_bytes(url: str, timeout: int = 20, referer: str | None = None):
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/*;q=0.8,*/*;q=0.7",
    }
    if referer:
        headers["Referer"] = referer
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        return response.read(), content_type


def is_probably_image_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return path.endswith((".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"))


def extract_image_urls_from_html(base_url: str, html_bytes: bytes):
    html = unescape(html_bytes.decode("utf-8", errors="ignore"))
    candidates = []
    candidates.extend(OG_IMAGE_RE.findall(html))
    candidates.extend(TWITTER_IMAGE_RE.findall(html))
    candidates.extend(IMG_TAG_RE.findall(html))

    normalized = []
    seen = set()
    for candidate in candidates:
        url = urljoin(base_url, candidate.strip())
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if url not in seen:
            seen.add(url)
            normalized.append(url)
    return normalized


def extract_page_links_from_html(base_url: str, html_bytes: bytes):
    html = unescape(html_bytes.decode("utf-8", errors="ignore"))
    candidates = ANCHOR_TAG_RE.findall(html)

    normalized = []
    seen = set()
    for candidate in candidates:
        url = urljoin(base_url, candidate.strip())
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            continue
        if url not in seen:
            seen.add(url)
            normalized.append(url)
    return normalized


def sniff_extension(data: bytes, fallback: str = ".jpg") -> str:
    detected = imghdr.what(None, h=data)
    mapping = {
        "jpeg": ".jpg",
        "png": ".png",
        "webp": ".webp",
        "bmp": ".bmp",
        "gif": ".gif",
    }
    return mapping.get(detected, fallback)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_url_list(input_path: Path):
    lines = input_path.read_text(encoding="utf-8-sig").splitlines()
    urls = []
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def read_json_config(config_path: Path) -> dict:
    return json.loads(config_path.read_text(encoding="utf-8-sig"))


def save_image(data: bytes, output_dir: Path, digest: str) -> Path:
    ext = sniff_extension(data)
    path = output_dir / f"{digest[:16]}{ext}"
    path.write_bytes(data)
    return path


def image_size_ok(data: bytes, min_size: int) -> bool:
    if not min_size:
        return True
    try:
        from io import BytesIO
        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            return width >= min_size and height >= min_size
    except Exception:
        return True


def get_image_dimensions(data: bytes):
    try:
        from io import BytesIO
        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            return str(width), str(height)
    except Exception:
        return "", ""


def ensure_metadata_file(path: Path) -> None:
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        handle.write("saved_path,source_page,image_url,sha256,width,height\n")


def append_metadata(path: Path, saved_path: str, source_page: str, image_url: str, digest: str, width: str, height: str) -> None:
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([saved_path, source_page, image_url, digest, width, height])


def load_seen_hashes(metadata_path: Path):
    seen = set()
    if not metadata_path.exists():
        return seen
    for line in metadata_path.read_text(encoding="utf-8-sig").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) >= 4:
            seen.add(parts[3])
    return seen


def domain_allowed(url: str, allowed_domains: set[str]) -> bool:
    if not allowed_domains:
        return True
    host = (urlparse(url).hostname or "").lower()
    return any(host == domain or host.endswith(f".{domain}") for domain in allowed_domains)


def collect_seed_urls(seeds, delay: float, max_depth: int, allowed_domains: set[str], same_domain_only: bool):
    if max_depth <= 0:
        return seeds

    queue = [(url, 0, (urlparse(url).hostname or "").lower()) for url in seeds]
    seen_pages = set()
    collected = []

    while queue:
        current_url, depth, root_host = queue.pop(0)
        if current_url in seen_pages:
            continue
        seen_pages.add(current_url)
        collected.append(current_url)

        if depth >= max_depth:
            continue
        if not domain_allowed(current_url, allowed_domains):
            continue

        time.sleep(delay)
        try:
            payload, content_type = fetch_bytes(current_url)
        except (HTTPError, URLError, TimeoutError):
            continue

        if content_type.startswith("image/") or is_probably_image_url(current_url):
            continue

        for next_url in extract_page_links_from_html(current_url, payload):
            next_host = (urlparse(next_url).hostname or "").lower()
            if same_domain_only and root_host and next_host != root_host:
                continue
            if not domain_allowed(next_url, allowed_domains):
                continue
            if next_url not in seen_pages:
                queue.append((next_url, depth + 1, root_host))

    return collected


def main() -> None:
    parser = argparse.ArgumentParser(description="Download images from authorized pages or direct image URLs.")
    parser.add_argument("--input", help="Text file with one page URL or image URL per line.")
    parser.add_argument("--config", help="JSON config file with crawler options.")
    parser.add_argument("--output", required=True, help="Directory for downloaded images.")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between requests.")
    parser.add_argument("--min-size", type=int, default=0, help="Skip images smaller than this many pixels on either side.")
    parser.add_argument("--max-depth", type=int, default=0, help="Crawl page links up to this depth.")
    parser.add_argument("--same-domain-only", action="store_true", help="When crawling, stay on the seed URL's domain.")
    parser.add_argument("--allow-domain", action="append", default=[], help="Allowed domain filter. Repeat for multiple values.")
    parser.add_argument("--image-regex", help="Only save image URLs matching this regex.")
    args = parser.parse_args()

    if not args.input and not args.config:
        parser.error("one of --input or --config is required")

    config = {}
    if args.config:
        config = read_json_config(Path(args.config))

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "metadata.csv"
    ensure_metadata_file(metadata_path)
    seen_hashes = load_seen_hashes(metadata_path)

    urls = []
    if args.input:
        urls.extend(read_url_list(Path(args.input)))
    urls.extend(config.get("seeds", []))

    allowed_domains = {domain.lower() for domain in args.allow_domain}
    allowed_domains.update(domain.lower() for domain in config.get("allow_domains", []))
    max_depth = config.get("max_depth", args.max_depth)
    same_domain_only = config.get("same_domain_only", args.same_domain_only)
    regex_pattern = config.get("image_regex") or args.image_regex
    image_regex = re.compile(regex_pattern) if regex_pattern else None

    urls = collect_seed_urls(
        seeds=urls,
        delay=args.delay,
        max_depth=max_depth,
        allowed_domains=allowed_domains,
        same_domain_only=same_domain_only,
    )
    print(f"Loaded {len(urls)} seed URLs")

    for index, seed_url in enumerate(urls, start=1):
        print(f"[{index}/{len(urls)}] Processing {seed_url}")
        time.sleep(args.delay)
        try:
            payload, content_type = fetch_bytes(seed_url)
        except (HTTPError, URLError, TimeoutError) as exc:
            print(f"  Failed to fetch seed URL: {exc}")
            continue

        if is_probably_image_url(seed_url) or content_type.startswith("image/"):
            image_urls = [seed_url]
            image_payloads = {seed_url: payload}
            page_url = seed_url
        else:
            image_urls = extract_image_urls_from_html(seed_url, payload)
            image_payloads = {}
            page_url = seed_url

        if not image_urls:
            print("  No image URLs found")
            continue

        for image_url in image_urls:
            time.sleep(args.delay)
            try:
                if image_regex and not image_regex.search(image_url):
                    print(f"  Regex skipped: {image_url}")
                    continue
                if not domain_allowed(image_url, allowed_domains):
                    print(f"  Domain skipped: {image_url}")
                    continue

                image_data = image_payloads.get(image_url)
                if image_data is None:
                    image_data, image_type = fetch_bytes(image_url, referer=page_url)
                    if not image_type.startswith("image/") and not is_probably_image_url(image_url):
                        print(f"  Skipped non-image URL: {image_url}")
                        continue

                digest = sha256_bytes(image_data)
                if digest in seen_hashes:
                    print(f"  Duplicate skipped: {image_url}")
                    continue
                if not image_size_ok(image_data, args.min_size):
                    print(f"  Too small skipped: {image_url}")
                    continue

                saved_path = save_image(image_data, output_dir, digest)
                width, height = get_image_dimensions(image_data)
                seen_hashes.add(digest)
                append_metadata(metadata_path, str(saved_path), page_url, image_url, digest, width, height)
                print(f"  Saved {saved_path.name}")
            except (HTTPError, URLError, TimeoutError) as exc:
                print(f"  Failed image URL: {exc}")
            except Exception as exc:
                print(f"  Unexpected error: {exc}")


if __name__ == "__main__":
    main()

