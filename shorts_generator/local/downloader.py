"""Local YouTube download via yt-dlp.

Returns a local mp4 path so the rest of the local pipeline can read it
directly off disk.
"""
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlparse
from typing import Optional


def _import_ytdlp():
    try:
        import yt_dlp  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "yt-dlp is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e
    return yt_dlp


def _format_for(fmt: str) -> str:
    """Map our '720' / '1080' shorthand to a yt-dlp format selector."""
    try:
        height = int(fmt)
    except ValueError:
        height = 720
    return (
        f"bestvideo[height<={height}][ext=mp4]+bestaudio[ext=m4a]/"
        f"best[height<={height}][ext=mp4]/best"
    )


def _resolve_local_path(source: str) -> Optional[str]:
    """Return a local filesystem path if the input already points at one."""
    parsed = urlparse(source)
    if parsed.scheme == "file":
        raw_path = unquote(parsed.path)
        if parsed.netloc and parsed.netloc not in ("", "localhost"):
            raw_path = f"//{parsed.netloc}{raw_path}"
        candidate = Path(raw_path).expanduser()
        if candidate.exists() and candidate.is_file():
            return str(candidate.resolve())
        raise RuntimeError(f"Local file URL does not exist: {source}")

    if parsed.scheme in ("http", "https"):
        return None

    candidate = Path(source).expanduser()
    if candidate.exists() and candidate.is_file():
        return str(candidate.resolve())

    if any(sep in source for sep in (os.sep, "/")) or source.startswith("~") or source.startswith("."):
        raise RuntimeError(f"Local file path does not exist: {source}")

    return None


def _ensure_mp4_at(src: str, dest: str) -> None:
    """Make sure `dest` exists as an mp4 copy of `src` (remux if needed)."""
    if os.path.abspath(src) == os.path.abspath(dest):
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if src.lower().endswith(".mp4"):
        shutil.copyfile(src, dest)
        return
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-c", "copy", dest],
        check=True,
    )


def download_youtube_local(video_url: str, target_path: str, fmt: str = "720") -> str:
    """Resolve video_url (YouTube URL, other remote URL, local path, or
    file:// URL) into a local mp4 at exactly `target_path`."""
    local_path = _resolve_local_path(video_url)
    if local_path:
        print(f"[download/local] using local file: {local_path} -> {target_path}", flush=True)
        _ensure_mp4_at(local_path, target_path)
        return target_path

    yt_dlp = _import_ytdlp()
    os.makedirs(os.path.dirname(target_path), exist_ok=True)

    print(f"[download/local] {video_url} @ {fmt}p -> {target_path}", flush=True)
    ydl_opts = {
        "format": _format_for(fmt),
        "outtmpl": target_path + ".download.%(ext)s",
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        downloaded = ydl.prepare_filename(info)
        # merge_output_format may rename the extension after merge
        if not os.path.exists(downloaded):
            stem, _ = os.path.splitext(downloaded)
            for ext in (".mp4", ".mkv", ".webm"):
                if os.path.exists(stem + ext):
                    downloaded = stem + ext
                    break

    _ensure_mp4_at(downloaded, target_path)
    if os.path.abspath(downloaded) != os.path.abspath(target_path) and os.path.exists(downloaded):
        os.remove(downloaded)

    print(f"[download/local] ready: {target_path}", flush=True)
    return target_path
