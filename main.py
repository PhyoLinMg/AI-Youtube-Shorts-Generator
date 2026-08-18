"""CLI entry point.

Usage:
    python main.py "https://www.youtube.com/watch?v=..." \
        --num-clips 3 --aspect-ratio 9:16
"""
import argparse
import json
import sys

# Windows uses 'charmap' by default, which can't encode Unicode characters
# like →. Reconfigure stdout/stderr to UTF-8 so output works on all platforms.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from shorts_generator import generate_chapters, generate_shorts, generate_threads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI YouTube Shorts Generator")
    parser.add_argument(
        "url", nargs="?", default=None,
        help="YouTube URL, file:// URL, or local file path. For --clip-type "
             "thread, this is episode A -- pair it with --url-b for episode B.",
    )
    parser.add_argument(
        "--mode",
        choices=["api", "local"],
        default="api",
        help="api (default, MuAPI) or local (remote URL, file://, or local path + faster-whisper + LLM provider + ffmpeg).",
    )
    parser.add_argument("--num-clips", type=int, default=3, help="How many shorts to render (default: 3)")
    parser.add_argument("--aspect-ratio", default="9:16", help="Output aspect ratio (default: 9:16)")
    parser.add_argument("--format", default="1080", help="Source download resolution: 360 / 480 / 720 / 1080 (default: 1080)")
    parser.add_argument("--language", default=None, help="Force Whisper language code, e.g. 'en' (default: auto-detect)")
    parser.add_argument("--output-json", default=None, help="Write the full result JSON to this path")
    parser.add_argument(
        "--no-captions",
        dest="captions",
        action="store_false",
        default=True,
        help="Disable fade-in caption burn-in (captions are on by default in both modes).",
    )
    parser.add_argument(
        "--caption-fade-duration",
        type=float,
        default=0.3,
        help="Caption fade-in duration in seconds (default: 0.3)",
    )
    parser.add_argument(
        "--no-word-highlight",
        dest="word_highlight",
        action="store_false",
        default=True,
        help="Disable per-word highlight animation; caption shows a plain fading phrase instead.",
    )
    parser.add_argument(
        "--no-hook-card",
        dest="hook_card",
        action="store_false",
        default=True,
        help="Disable the opening hook-card overlay (bold on-screen hook text over the "
             "live footage for the first 1.5s; on by default).",
    )
    parser.add_argument(
        "--end-card",
        dest="end_card",
        action="store_true",
        default=False,
        help="Enable the closing end-card overlay (bold on-screen follow-up CTA text over the "
             "live footage for the last ~2s; off by default).",
    )
    parser.add_argument(
        "--framing",
        choices=["locked", "adaptive"],
        default="locked",
        help="locked (default): static speaker-centered crop. adaptive: cursor/person-aware crop "
             "for screen-recording content that alternates between facecam and screen activity "
             "(--mode local only).",
    )
    parser.add_argument(
        "--filename-style",
        choices=["specific", "generic"],
        default=None,
        help="specific (default): slugified highlight title, e.g. My_Big_Moment.mp4. "
             "generic: positional, video1.mp4, video2.mp4, ... "
             "Falls back to the SHORT_FILENAME_STYLE env var when unset.",
    )
    parser.add_argument(
        "--url-b",
        default=None,
        help="Second episode URL, required together with the positional url "
             "when --clip-type thread (paired as episode A + episode B).",
    )
    parser.add_argument(
        "--clip-type",
        choices=["shorts", "chapters", "thread"],
        default="shorts",
        help="shorts (default): viral 9:16 Shorts. chapters: long-form landscape "
             "chapter cuts, up to 15min each, full topic context, --mode local only. "
             "thread: up to --num-clips same-topic compilations built from the "
             "positional url (episode A) and --url-b (episode B), captions only, "
             "no video download.",
    )
    parser.add_argument(
        "--num-chapters",
        type=int,
        default=5,
        help="Target chapter count for --clip-type chapters (default: 5); the model "
             "may return 3-8 based on natural topic boundaries.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.clip_type == "thread":
        if not args.url or not args.url_b:
            print("\nFAILED: --clip-type thread requires both the positional url (episode A) and --url-b (episode B)", file=sys.stderr)
            return 1
    elif not args.url:
        print("\nFAILED: url is required for --clip-type shorts/chapters", file=sys.stderr)
        return 1

    if args.clip_type == "thread":
        # generate_threads() ingests url/url_b caption-only and uses
        # num_clips -- every other shorts/chapters-only flag below is
        # silently discarded, so tell the user which ones (if any) they
        # explicitly passed but that won't do anything here.
        thread_mode_explicit = any(a == "--mode" or a.startswith("--mode=") for a in sys.argv)
        ignored_flags = []
        if thread_mode_explicit:
            ignored_flags.append(f"--mode {args.mode}")
        if args.aspect_ratio != "9:16":
            ignored_flags.append(f"--aspect-ratio {args.aspect_ratio}")
        if args.format != "1080":
            ignored_flags.append(f"--format {args.format}")
        if args.language is not None:
            ignored_flags.append(f"--language {args.language}")
        if args.framing != "locked":
            ignored_flags.append(f"--framing {args.framing}")
        if args.filename_style is not None:
            ignored_flags.append(f"--filename-style {args.filename_style}")
        if args.captions is False:
            ignored_flags.append("--no-captions")
        if args.caption_fade_duration != 0.3:
            ignored_flags.append(f"--caption-fade-duration {args.caption_fade_duration}")
        if args.word_highlight is False:
            ignored_flags.append("--no-word-highlight")
        if args.hook_card is False:
            ignored_flags.append("--no-hook-card")
        if args.end_card is True:
            ignored_flags.append("--end-card")
        if args.num_chapters != 5:
            ignored_flags.append(f"--num-chapters {args.num_chapters}")
        if ignored_flags:
            print(
                f"[main] --clip-type thread ignores: {', '.join(ignored_flags)} "
                "(captions are fetched from YouTube directly, no download/transcribe step)",
                file=sys.stderr,
            )

    if args.clip_type == "chapters":
        # Only warn if --mode was explicitly typed and isn't "local" -- args.mode
        # defaults to "api" when omitted, and the plain, most natural invocation
        # (`--clip-type chapters` with no --mode at all) must not spuriously warn
        # on every single run just because the shorts-path default happens to
        # differ from what chapters always uses anyway.
        if "--mode" in sys.argv and args.mode != "local":
            print(f"[main] --clip-type chapters is local-only; ignoring --mode {args.mode!r} and using local", file=sys.stderr)
        ignored_flags = []
        if args.aspect_ratio != "9:16":
            ignored_flags.append(f"--aspect-ratio {args.aspect_ratio}")
        if args.framing != "locked":
            ignored_flags.append(f"--framing {args.framing}")
        if args.hook_card is False:
            ignored_flags.append("--no-hook-card")
        if args.end_card is True:
            ignored_flags.append("--end-card")
        if args.num_clips != 3:
            ignored_flags.append(f"--num-clips {args.num_clips}")
        if args.filename_style is not None:
            ignored_flags.append(f"--filename-style {args.filename_style}")
        if ignored_flags:
            print(
                f"[main] --clip-type chapters ignores: {', '.join(ignored_flags)} "
                "(no crop, no card overlays in this path)",
                file=sys.stderr,
            )

    try:
        if args.clip_type == "thread":
            result = generate_threads(url_a=args.url, url_b=args.url_b, num_clips=args.num_clips)
        elif args.clip_type == "chapters":
            result = generate_chapters(
                youtube_url=args.url,
                num_chapters=args.num_chapters,
                download_format=args.format,
                language=args.language,
                captions=args.captions,
                caption_fade_duration=args.caption_fade_duration,
                word_highlight=args.word_highlight,
            )
        else:
            result = generate_shorts(
                youtube_url=args.url,
                num_clips=args.num_clips,
                aspect_ratio=args.aspect_ratio,
                download_format=args.format,
                language=args.language,
                mode=args.mode,
                captions=args.captions,
                caption_fade_duration=args.caption_fade_duration,
                word_highlight=args.word_highlight,
                framing=args.framing,
                hook_card=args.hook_card,
                end_card=args.end_card,
                filename_style=args.filename_style,
            )
    except Exception as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1

    if args.clip_type == "thread" and not result:
        print("\nNo shared-question thread found between these two episodes -- nothing to build.", file=sys.stderr)
        print(
            "This can happen if the two episodes don't genuinely answer the same question -- "
            "try a different pair, or lower --num-clips.",
            file=sys.stderr,
        )
        return 1

    print("\n" + "=" * 72)
    if args.clip_type == "thread":
        print(f"Threads built:   {len(result)} (requested {args.num_clips})")
        for i, t in enumerate(result, 1):
            print(f"\n#{i}  {t.get('shared_question')}")
            print(f"     title:   {t.get('title')}")
            print(f"     desc:    {t.get('description')}")
            print(f"     thesis:  {t.get('thesis')}")
            print(f"     bridge:  {t.get('bridge')}")
            ea, eb = t["episode_a"], t["episode_b"]
            print(f"     episode A: {ea['title']} ({ea['start_time']:.1f}s -> {ea['end_time']:.1f}s)")
            print(f"     episode B: {eb['title']} ({eb['start_time']:.1f}s -> {eb['end_time']:.1f}s)")
            print(f"     clip:      {t.get('clip_url')}")
    elif args.clip_type == "chapters":
        print(f"Output folder: {result.get('output_dir')}")
        print(f"Source video:  {result['source_video_url']}")
        print(f"Chapters:      {len(result['chapters'])} produced (target was {args.num_chapters})")
        print("=" * 72)
        for i, c in enumerate(result["chapters"], 1):
            print(f"\n#{i}  {c.get('start_time'):.1f}s -> {c.get('end_time'):.1f}s")
            print(f"     title:   {c.get('title')}")
            if c.get("summary"):
                print(f"     summary: {c.get('summary')}")
            if c.get("clip_url"):
                print(f"     clip:    {c['clip_url']}")
            else:
                print(f"     clip:    FAILED ({c.get('error')})")
    else:
        print(f"Mode:          {result.get('mode', args.mode)}")
        print(f"Output folder: {result.get('output_dir')}")
        print(f"Source video:  {result['source_video_url']}")
        print(f"Highlights:    {len(result['highlights'])} candidates → kept top {len(result['shorts'])}")
        print("=" * 72)
        for i, s in enumerate(result["shorts"], 1):
            print(f"\n#{i}  score={s.get('score')}  {s.get('start_time'):.1f}s → {s.get('end_time'):.1f}s")
            print(f"     title:  {s.get('yt_title') or s.get('title')}")
            print(f"     hook:   {s.get('hook_sentence')}")
            if s.get("description"):
                print(f"     desc:   {s.get('description')}")
            if s.get("yt_hashtags"):
                print(f"     tags:   {' '.join(s.get('yt_hashtags'))}")
            if s.get("clip_url"):
                print(f"     clip:   {s['clip_url']}")
            else:
                print(f"     clip:   FAILED ({s.get('error')})")

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nFull JSON written to {args.output_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
