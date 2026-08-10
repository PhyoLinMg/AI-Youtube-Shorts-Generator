"""Add one or more YouTube videos to the local corpus from auto-captions
only -- no video/audio download (see shorts_generator/local/caption_ingest.py).
Use this to grow the corpus for --clip-type thread without paying the full
download+Whisper cost per episode up front.

Usage:
    python ingest_corpus.py <youtube_url> [<youtube_url> ...]
"""
import sys

from shorts_generator.local.caption_ingest import ingest_captions


def main() -> int:
    urls = sys.argv[1:]
    if not urls:
        print("Usage: python ingest_corpus.py <youtube_url> [<youtube_url> ...]", file=sys.stderr)
        return 1

    failed = False
    for url in urls:
        try:
            result = ingest_captions(url)
            print(f"OK  {url} -> {result['run_dir']} ({result['segment_count']} segments, {result['duration']:.0f}s)")
        except Exception as e:
            failed = True
            print(f"FAIL {url}: {e}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
