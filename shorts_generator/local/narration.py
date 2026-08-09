"""ElevenLabs narrator voice for thread bridges -- see thread_builder.py for
where "thesis" and "bridge" text comes from. Renders each line as audio via
ElevenLabs, then composites it onto a plain title card matching the
channel's existing hook-card typography (Anton font, white text on a
translucent black box -- see hook_card.py) so it drops into the same
ffmpeg-concat assembly as the live-footage clips (see thread_assembler.py).
"""
import os
import subprocess
from typing import Type

from ..config import ELEVENLABS_VOICE_ID, require_elevenlabs_key
from ..hook_card import FONT_PATH

DEFAULT_VOICE_ID = ELEVENLABS_VOICE_ID
CARD_WIDTH = 1080
CARD_HEIGHT = 1920
CARD_FPS = "30000/1001"
CARD_BG_COLOR = "0x0d0d0d"


class NarrationError(RuntimeError):
    """Raised when ElevenLabs synthesis or card rendering fails."""


def _get_elevenlabs_client_class() -> Type:
    try:
        from elevenlabs.client import ElevenLabs
    except ImportError as e:
        raise NarrationError(
            "elevenlabs is required for thread narration. Install it with:\n"
            "    pip install elevenlabs"
        ) from e
    return ElevenLabs


def synthesize_narration(text: str, out_path: str, voice_id: str = DEFAULT_VOICE_ID) -> str:
    """Call ElevenLabs TTS and write the audio to out_path (mp3)."""
    client_cls = _get_elevenlabs_client_class()
    client = client_cls(api_key=require_elevenlabs_key())
    try:
        audio = client.text_to_speech.convert(
            voice_id=voice_id,
            text=text,
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128",
        )
        with open(out_path, "wb") as f:
            for chunk in audio:
                f.write(chunk)
    except Exception as e:
        raise NarrationError(f"ElevenLabs synthesis failed: {e}") from e
    return out_path


def _wrap_text(text: str, max_chars_per_line: int = 28) -> str:
    words = text.split()
    lines = []
    current: list = []
    current_len = 0
    for word in words:
        added_len = len(word) + (1 if current else 0)
        if current and current_len + added_len > max_chars_per_line:
            lines.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += added_len
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def render_narration_card(audio_path: str, text: str, out_path: str) -> str:
    """Composite `text` (Anton font, boxed, auto-wrapped) over a plain dark
    card, muxed with the narration audio at audio_path, sized/timed to match
    the thread's live-footage clips (see thread_assembler.py's TARGET_*)."""
    text_file = out_path + ".txt"

    try:
        with open(text_file, "w", encoding="utf-8") as f:
            f.write(_wrap_text(text))

        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", audio_path],
            capture_output=True, text=True, check=True,
        )
        duration = float(probe.stdout.strip())

        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "lavfi", "-i", f"color=c={CARD_BG_COLOR}:s={CARD_WIDTH}x{CARD_HEIGHT}:r={CARD_FPS}:d={duration}",
                "-i", audio_path,
                "-vf",
                f"drawtext=fontfile='{FONT_PATH}':textfile='{text_file}':fontsize=64:fontcolor=white:"
                "box=1:boxcolor=black@0.55:boxborderw=16:x=(w-text_w)/2:y=(h-text_h)/2:line_spacing=20:"
                "expansion=none",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", CARD_FPS,
                "-c:a", "aac", "-ar", "44100", "-ac", "2", "-b:a", "192k",
                "-shortest", out_path,
            ],
            check=True, capture_output=True, text=True,
        )
    except subprocess.CalledProcessError as e:
        raise NarrationError(f"narration card render failed: {e.stderr}") from e
    except (OSError, ValueError) as e:
        raise NarrationError(f"narration card render failed: {e}") from e
    finally:
        if os.path.exists(text_file):
            os.remove(text_file)
    return out_path
