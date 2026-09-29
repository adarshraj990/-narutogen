"""
====================================================================================================
MODULE 1: KOKORO-82M LOCAL TTS GENERATOR
====================================================================================================
Converts subtitle cues (.srt) into natural-sounding, studio-quality base audio files (.wav) locally.
Features:
- 100% Free, NO API keys, NO cloud subscriptions, runs entirely local / offline.
- Powered by Kokoro-82M: State-of-the-art lightweight (82M parameter) neural text-to-speech.
- Multi-Language Support: Hindi, Spanish, French, and Portuguese.
- Smooth, non-blocking execution with in-memory pipeline caching and live progress reporting.
====================================================================================================
"""

import os
import re
import sys
import shutil
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import numpy as np

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    soundfile = None  # type: ignore
    SOUNDFILE_AVAILABLE = False

try:
    import pysrt
    PYSRT_AVAILABLE = True
except ImportError:
    pysrt = None  # type: ignore
    PYSRT_AVAILABLE = False

# Try importing Kokoro KPipeline gracefully
KOKORO_AVAILABLE = False
KPipeline = None  # type: ignore

try:
    from kokoro import KPipeline  # type: ignore
    KOKORO_AVAILABLE = True
except ImportError:
    KPipeline = None
    KOKORO_AVAILABLE = False

# ==================================================================================================
# KOKORO MULTI-LANGUAGE VOICES CATALOG
# ==================================================================================================
# lang_code mapping: 'h' = Hindi, 'e' = Spanish, 'f' = French, 'p' = Portuguese, 'a' = English
KOKORO_VOICES = {
    # Hindi (Primary)
    "hi_omega": {"lang_code": "h", "voice": "hm_omega", "name": "Hindi Male (Omega)"},
    "hi_alpha": {"lang_code": "h", "voice": "hf_alpha", "name": "Hindi Female (Alpha)"},
    "hi_psi": {"lang_code": "h", "voice": "hm_psi", "name": "Hindi Male (Psi)"},
    "hi_beta": {"lang_code": "h", "voice": "hf_beta", "name": "Hindi Female (Beta)"},
    "hindi": {"lang_code": "h", "voice": "hm_omega", "name": "Hindi Male (Default)"},
    "hi": {"lang_code": "h", "voice": "hm_omega", "name": "Hindi Male (Default)"},

    # Spanish
    "es_alex": {"lang_code": "e", "voice": "em_alex", "name": "Spanish Male (Alex)"},
    "es_dora": {"lang_code": "e", "voice": "ef_dora", "name": "Spanish Female (Dora)"},
    "es_santa": {"lang_code": "e", "voice": "em_santa", "name": "Spanish Male (Santa)"},
    "spanish": {"lang_code": "e", "voice": "em_alex", "name": "Spanish Male (Default)"},
    "es": {"lang_code": "e", "voice": "em_alex", "name": "Spanish Male (Default)"},

    # French
    "fr_siwis": {"lang_code": "f", "voice": "ff_siwis", "name": "French Female (Siwis)"},
    "french": {"lang_code": "f", "voice": "ff_siwis", "name": "French Female (Default)"},
    "fr": {"lang_code": "f", "voice": "ff_siwis", "name": "French Female (Default)"},

    # Portuguese (Brazil)
    "pt_alex": {"lang_code": "p", "voice": "pm_alex", "name": "Portuguese Male (Alex)"},
    "pt_dora": {"lang_code": "p", "voice": "pf_dora", "name": "Portuguese Female (Dora)"},
    "pt_santa": {"lang_code": "p", "voice": "pm_santa", "name": "Portuguese Male (Santa)"},
    "portuguese": {"lang_code": "p", "voice": "pm_alex", "name": "Portuguese Male (Default)"},
    "pt": {"lang_code": "p", "voice": "pm_alex", "name": "Portuguese Male (Default)"},
    "pt-br": {"lang_code": "p", "voice": "pm_alex", "name": "Portuguese Male (Default)"},
}

DEFAULT_LANG_CODE = "h"
DEFAULT_VOICE = "hm_omega"
KOKORO_SAMPLE_RATE = 24000

# Global cache of initialized KPipeline instances
_PIPELINE_CACHE: Dict[str, Any] = {}


def get_kokoro_pipeline(lang_code: str = "h") -> Any:
    """Returns or initializes a cached KPipeline instance for the given language code."""
    global _PIPELINE_CACHE, KPipeline, KOKORO_AVAILABLE

    if not KOKORO_AVAILABLE or KPipeline is None:
        try:
            from kokoro import KPipeline as _KP  # type: ignore
            KPipeline = _KP
            KOKORO_AVAILABLE = True
        except ImportError:
            raise RuntimeError(
                "Kokoro-82M TTS is not installed in the environment.\n"
                "Install it with: pip install kokoro soundfile"
            )

    lang_code = lang_code.lower()
    if lang_code not in _PIPELINE_CACHE:
        print(f"📦 [KOKORO TTS] Initializing Kokoro-82M pipeline for language code '{lang_code}'...")
        _PIPELINE_CACHE[lang_code] = KPipeline(lang_code=lang_code)
        print(f"✅ [KOKORO TTS] Pipeline ready for '{lang_code}'.")

    return _PIPELINE_CACHE[lang_code]


def resolve_kokoro_voice(voice_or_lang: str) -> Tuple[str, str]:
    """Resolves voice input into (lang_code, voice_name)."""
    key = str(voice_or_lang).strip().lower()
    if key in KOKORO_VOICES:
        item = KOKORO_VOICES[key]
        return item["lang_code"], item["voice"]

    # Match by prefix (e.g., 'h' for Hindi, 'e' for Spanish, 'f' for French, 'p' for Portuguese)
    if key.startswith("h"):
        return "h", "hm_omega"
    elif key.startswith("es") or key.startswith("e"):
        return "e", "em_alex"
    elif key.startswith("fr") or key.startswith("f"):
        return "f", "ff_siwis"
    elif key.startswith("pt") or key.startswith("p"):
        return "p", "pm_alex"

    return DEFAULT_LANG_CODE, DEFAULT_VOICE


class SubtitleCue:
    """Data representation of a single subtitle line."""
    def __init__(self, cue_id: int, start_ms: int, end_ms: int, text: str):
        self.cue_id = cue_id
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.target_dur_ms = max(200, end_ms - start_ms)
        self.text = text
        self.base_wav_path: Optional[str] = None
        self.rvc_wav_path: Optional[str] = None

    def __repr__(self):
        return f"<SubtitleCue #{self.cue_id} [{self.start_ms}ms -> {self.end_ms}ms ({self.target_dur_ms}ms)]: '{self.text[:25]}...'>"


def clean_dialogue_text(text: str) -> str:
    """Cleans dialogue text by stripping HTML tags, subtitle formatting, and sound effect annotations."""
    if not text:
        return ""
    # Strip HTML tags
    clean = re.sub(r"<[^>]+>", "", text)
    # Strip sound effect annotations e.g. [laughs], (music)
    clean = re.sub(r"\[.*?\]", "", clean)
    clean = re.sub(r"\(.*?\)", "", clean)
    # Strip ASS subtitle formatting tags
    clean = re.sub(r"\{.*?\}", "", clean)
    # Replace multiple whitespace
    clean = re.sub(r"\s+", " ", clean).strip()
    return clean


def parse_srt(srt_path: str) -> List[SubtitleCue]:
    """Parses SRT file into SubtitleCue objects with universal line-ending support."""
    cues = []
    if PYSRT_AVAILABLE and pysrt:
        try:
            subs = pysrt.open(srt_path, encoding="utf-8")
            for i, sub in enumerate(subs, start=1):
                clean_txt = clean_dialogue_text(sub.text)
                if not clean_txt:
                    continue
                start_ms = int(sub.start.ordinal)
                end_ms = int(sub.end.ordinal)
                if end_ms <= start_ms:
                    end_ms = start_ms + 1000
                cues.append(SubtitleCue(cue_id=i, start_ms=start_ms, end_ms=end_ms, text=clean_txt))
            if cues:
                return cues
        except Exception:
            pass

    # Pure-Python universal regex parser fallback
    with open(srt_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        content = f.read().replace("\r\n", "\n")

    blocks = re.split(r"\n\s*\n", content.strip())
    pattern = re.compile(
        r"(\d+)\s*\n\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n([\s\S]+)"
    )

    def _parse_ts(ts_str):
        ts_str = ts_str.strip().replace(",", ".")
        parts = ts_str.split(":")
        if len(parts) == 3:
            h = int(parts[0])
            m = int(parts[1])
            s_parts = parts[2].split(".")
            s = int(s_parts[0])
            ms = int(s_parts[1].ljust(3, "0")[:3]) if len(s_parts) > 1 else 0
            return ((h * 3600) + (m * 60) + s) * 1000 + ms
        return 0

    for block in blocks:
        match = pattern.search(block.strip())
        if match:
            idx = int(match.group(1))
            start_ms = _parse_ts(match.group(2))
            end_ms = _parse_ts(match.group(3))
            clean_txt = clean_dialogue_text(match.group(4))
            if clean_txt:
                if end_ms <= start_ms:
                    end_ms = start_ms + 1000
                cues.append(SubtitleCue(cue_id=idx, start_ms=start_ms, end_ms=end_ms, text=clean_txt))

    return cues


def synthesize_single_cue_kokoro(
    cue: SubtitleCue,
    pipeline: Any,
    voice: str,
    output_wav_path: str,
    speed: float = 1.0,
) -> str:
    """Synthesizes a single SubtitleCue using Kokoro-82M into a 24kHz WAV file."""
    audio_chunks = []
    try:
        for _, _, audio in pipeline(cue.text, voice=voice, speed=speed):
            if audio is not None and len(audio) > 0:
                if hasattr(audio, "cpu"):
                    audio = audio.cpu().numpy()
                audio_chunks.append(audio)

        if audio_chunks:
            full_audio = np.concatenate(audio_chunks)
        else:
            # Fallback silence matching target duration
            silence_samples = int(KOKORO_SAMPLE_RATE * (cue.target_dur_ms / 1000.0))
            full_audio = np.zeros(max(2400, silence_samples), dtype=np.float32)

        sf.write(output_wav_path, full_audio, KOKORO_SAMPLE_RATE)
        cue.base_wav_path = output_wav_path
        return output_wav_path

    except Exception as e:
        print(f"⚠️ [KOKORO CUE ERROR] Cue #{cue.cue_id} ('{cue.text[:20]}...'): {e}")
        # Write safety silence
        silence_samples = int(KOKORO_SAMPLE_RATE * (cue.target_dur_ms / 1000.0))
        full_audio = np.zeros(max(2400, silence_samples), dtype=np.float32)
        sf.write(output_wav_path, full_audio, KOKORO_SAMPLE_RATE)
        cue.base_wav_path = output_wav_path
        return output_wav_path


def batch_generate_kokoro_speech(
    cues: List[SubtitleCue],
    output_dir: Union[str, Path] = "./outputs/base_tts",
    lang_code: str = "h",
    voice: str = "hm_omega",
    progress_callback=None,
) -> List[SubtitleCue]:
    """
    Synthesizes a batch of SubtitleCue objects using local Kokoro-82M.
    Fast, offline, and non-blocking.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(cues)
    if total == 0:
        return []

    print(f"\n⚡ [KOKORO BATCH TTS] Synthesizing {total} cues (Lang: '{lang_code}', Voice: '{voice}')...")
    start_time = time.time()

    pipeline = get_kokoro_pipeline(lang_code=lang_code)

    for idx, cue in enumerate(cues, start=1):
        cue_wav = str(out_dir / f"cue_{cue.cue_id:04d}_kokoro.wav")
        synthesize_single_cue_kokoro(cue, pipeline, voice=voice, output_wav_path=cue_wav)

        if progress_callback and total > 0:
            progress_callback(idx / total, desc=f"⚡ [Kokoro TTS] Synthesized: {idx}/{total} cues...")

        if idx % 25 == 0 or idx == total:
            elapsed = time.time() - start_time
            rate = idx / max(0.001, elapsed)
            print(f"   ↳ [Kokoro Progress] {idx}/{total} synthesized ({rate:.1f} cues/sec)")

    total_time = time.time() - start_time
    print(f"✅ [KOKORO BATCH TTS] Completed {total} cues in {total_time:.2f}s ({total/max(0.001, total_time):.1f} cues/sec)!\n")
    return cues


# Backwards-compatibility alias for pipeline.py
def generate_base_speech(
    input_source: Union[str, Path, List[SubtitleCue]],
    output_dir: Union[str, Path] = "./outputs/base_tts",
    voice: str = DEFAULT_VOICE,
    concurrency: int = 10,
    progress_callback=None,
) -> List[SubtitleCue]:
    """Universal high-level entrypoint for Module 1 using Kokoro-82M."""
    if isinstance(input_source, list) and len(input_source) > 0 and isinstance(input_source[0], SubtitleCue):
        cues = input_source
    else:
        cues = parse_srt(str(input_source))

    lang_code, voice_name = resolve_kokoro_voice(voice)
    return batch_generate_kokoro_speech(
        cues=cues,
        output_dir=output_dir,
        lang_code=lang_code,
        voice=voice_name,
        progress_callback=progress_callback,
    )


# Compatibility aliases for legacy runners
parse_srt_file = parse_srt
SUPPORTED_VOICES = KOKORO_VOICES
resolve_voice_name = resolve_kokoro_voice


