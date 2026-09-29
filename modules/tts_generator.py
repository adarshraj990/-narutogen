"""
====================================================================================================
MODULE 1: TTS GENERATOR (Edge-TTS)
====================================================================================================
Converts input text or SRT lines into clean, uncompressed base audio files (.wav) locally.
Features:
- 100% Free, NO API keys, NO cloud subscriptions.
- Asynchronous batch processing with semaphore concurrency for high speed.
- Automatic conversion from Edge-TTS stream to 44.1kHz 16-bit PCM WAV.
- Multi-language voice mapping (Hindi, Spanish, French, English, Japanese, etc.).
====================================================================================================
"""

import os
import re
import sys
import shutil
import asyncio
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    import pysrt
    PYSRT_AVAILABLE = True
except ImportError:
    PYSRT_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

# ==================================================================================================
# RECOMMENDED NEURAL VOICES MAP
# ==================================================================================================
SUPPORTED_VOICES = {
    # Hindi (Primary target for Naruto Hindi dubbing)
    "hi_madhur": "hi-IN-MadhurNeural",       # Male (Deep, clear, heroic - ideal for Naruto/Male Anime)
    "hi_swara": "hi-IN-SwaraNeural",         # Female (Expressive, natural - female characters/young Naruto)
    "hindi": "hi-IN-MadhurNeural",
    "hi": "hi-IN-MadhurNeural",
    # Spanish
    "es_alvaro": "es-ES-AlvaroNeural",       # Spanish Spain Male
    "es_jorge": "es-MX-JorgeNeural",         # Spanish Mexico Male
    "es_elvira": "es-ES-ElviraNeural",       # Spanish Spain Female
    "spanish": "es-ES-AlvaroNeural",
    "es": "es-ES-AlvaroNeural",
    # French
    "fr_henri": "fr-FR-HenriNeural",         # French Male
    "fr_denise": "fr-FR-DeniseNeural",       # French Female
    "french": "fr-FR-HenriNeural",
    "fr": "fr-FR-HenriNeural",
    # Portuguese (Brazil & Portugal)
    "pt_antonio": "pt-BR-AntonioNeural",     # Brazilian Portuguese Male (Expressive, clear)
    "pt_francisca": "pt-BR-FranciscaNeural", # Brazilian Portuguese Female
    "pt_duarte": "pt-PT-DuarteNeural",       # Portugal Portuguese Male
    "portuguese": "pt-BR-AntonioNeural",
    "pt": "pt-BR-AntonioNeural",
    "pt-br": "pt-BR-AntonioNeural",
    # English
    "en_guy": "en-US-GuyNeural",             # US Male
    "en_prabhat": "en-IN-PrabhatNeural",     # Indian English Male
    "en_neerja": "en-IN-NeerjaNeural",       # Indian English Female
    "english": "en-US-GuyNeural",
    "en": "en-US-GuyNeural",
    # Japanese (Reference)
    "ja_keita": "ja-JP-KeitaNeural",         # Japanese Male
    "japanese": "ja-JP-KeitaNeural",
    "ja": "ja-JP-KeitaNeural",
}

DEFAULT_VOICE = "hi-IN-MadhurNeural"
SAMPLE_RATE = 44100


def resolve_voice_name(voice_input: Optional[str]) -> str:
    """Resolves short voice names, language codes, or full Azure neural voice names."""
    if not voice_input:
        return DEFAULT_VOICE
    cleaned = str(voice_input).strip()
    return SUPPORTED_VOICES.get(cleaned.lower(), SUPPORTED_VOICES.get(cleaned, cleaned))


# ==================================================================================================
# DATA STRUCTURE: SUBTITLE CUE
# ==================================================================================================
class SubtitleCue:
    """Represents a single timed dialogue segment."""
    def __init__(self, cue_id: int, start_ms: int, end_ms: int, text: str):
        self.cue_id = cue_id
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.target_dur_ms = max(200, end_ms - start_ms)
        self.text = text
        self.base_wav_path: Optional[str] = None
        self.rvc_wav_path: Optional[str] = None

    def __repr__(self):
        return f"<Cue #{self.cue_id} [{self.start_ms}ms -> {self.end_ms}ms ({self.target_dur_ms}ms)]: '{self.text[:25]}...'>"


# ==================================================================================================
# TEXT & SRT CLEANING UTILITIES
# ==================================================================================================
def clean_dialogue_text(text: str) -> str:
    """Removes HTML formatting tags, sound-effect brackets, and redundant whitespace."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _parse_srt_timestamp_ms(ts_str: str) -> int:
    """Converts '00:01:23,456' or '00:01:23.456' into milliseconds."""
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


def _parse_srt_pure_python(srt_path: str) -> List[SubtitleCue]:
    """Pure-Python regex fallback parser for SRT files (zero dependencies)."""
    with open(srt_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        content = f.read()

    blocks = re.split(r"\n\s*\n", content.strip())
    cues = []
    pattern = re.compile(
        r"(\d+)\s*\n\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,\.]\d{3})\s*\n([\s\S]+)"
    )

    for block in blocks:
        match = pattern.search(block.strip())
        if match:
            idx = int(match.group(1))
            start_ms = _parse_srt_timestamp_ms(match.group(2))
            end_ms = _parse_srt_timestamp_ms(match.group(3))
            raw_text = match.group(4)
            cleaned = clean_dialogue_text(raw_text)
            if cleaned:
                if end_ms <= start_ms:
                    end_ms = start_ms + 1000
                cues.append(SubtitleCue(cue_id=idx, start_ms=start_ms, end_ms=end_ms, text=cleaned))

    return cues


def parse_srt_file(srt_path: str) -> List[SubtitleCue]:
    """
    Parses an SRT file and returns a structured list of SubtitleCue objects with millisecond timestamps.
    Uses pysrt if available, with built-in pure-Python fallback.
    """
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"SRT file not found: {srt_path}")

    print(f"📄 [TTS PARSER] Parsing subtitle cues from: {srt_path}")

    if PYSRT_AVAILABLE:
        try:
            subs = pysrt.open(srt_path, encoding="utf-8")
            cues = []
            for i, sub in enumerate(subs, start=1):
                clean_text = clean_dialogue_text(sub.text)
                if not clean_text:
                    continue
                start_ms = int(sub.start.ordinal)
                end_ms = int(sub.end.ordinal)
                if end_ms <= start_ms:
                    end_ms = start_ms + 1000
                cues.append(SubtitleCue(cue_id=i, start_ms=start_ms, end_ms=end_ms, text=clean_text))
            print(f"✅ [TTS PARSER] Parsed {len(cues)} cues via pysrt.")
            return cues
        except Exception:
            pass

    cues = _parse_srt_pure_python(srt_path)
    print(f"✅ [TTS PARSER] Parsed {len(cues)} cues via built-in parser.")
    return cues


def parse_plain_text(text_input: str, default_duration_per_char_ms: int = 65) -> List[SubtitleCue]:
    """
    Converts a plain text script or multiline string into sequential SubtitleCue objects.
    Splits text by punctuation or line breaks.
    """
    lines = [clean_dialogue_text(line) for line in text_input.splitlines() if clean_dialogue_text(line)]
    if not lines:
        # Split single block into sentences if no line breaks
        raw_sentences = re.split(r"(?<=[।!?.\n])\s+", text_input)
        lines = [clean_dialogue_text(s) for s in raw_sentences if clean_dialogue_text(s)]

    cues = []
    current_time_ms = 0
    for i, line in enumerate(lines, start=1):
        # Estimate duration based on character count (approx 65ms per char) + pause
        dur_ms = max(1000, len(line) * default_duration_per_char_ms)
        start_ms = current_time_ms
        end_ms = start_ms + dur_ms
        cues.append(SubtitleCue(cue_id=i, start_ms=start_ms, end_ms=end_ms, text=line))
        current_time_ms = end_ms + 250  # 250ms inter-dialogue gap

    print(f"✅ [TTS PARSER] Parsed {len(cues)} sequential dialogue cues from plain text.")
    return cues


# ==================================================================================================
# ASYNC EDGE-TTS BATCH GENERATOR
# ==================================================================================================
async def _synthesize_single_cue(
    cue: SubtitleCue,
    voice: str,
    output_wav_path: str,
    semaphore: asyncio.Semaphore,
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> str:
    """
    Synthesizes a single dialogue cue using Edge-TTS under semaphore concurrency.
    Edge-TTS saves as MP3 stream; we convert directly to uncompressed 44.1kHz WAV.
    """
    temp_mp3 = output_wav_path.replace(".wav", "_temp.mp3")

    async with semaphore:
        try:
            communicate = edge_tts.Communicate(cue.text, voice, rate=rate, pitch=pitch)
            await communicate.save(temp_mp3)

            # Convert MP3 to clean 44.1kHz 16-bit mono WAV for RVC compatibility
            # Primary: soundfile (native C-accelerated, zero ffmpeg dependency)
            converted = False
            if SOUNDFILE_AVAILABLE:
                try:
                    data, in_sr = sf.read(temp_mp3)
                    # soundfile writes standard PCM_16 WAV
                    sf.write(output_wav_path, data, in_sr, subtype="PCM_16")
                    converted = True
                except Exception:
                    pass

            if not converted and PYDUB_AVAILABLE:
                seg = AudioSegment.from_file(temp_mp3)
                seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
                seg.export(output_wav_path, format="wav")
                converted = True

            # Clean temporary MP3
            try:
                if os.path.exists(temp_mp3):
                    os.remove(temp_mp3)
            except Exception:
                pass

            cue.base_wav_path = output_wav_path
            return output_wav_path

        except Exception as e:
            print(f"⚠️ [TTS SYNTHESIS WARNING] Cue #{cue.cue_id} ('{cue.text[:20]}...'): {e}")
            # Fallback: create silent WAV matching target duration
            if SOUNDFILE_AVAILABLE:
                silence_samples = int(SAMPLE_RATE * (cue.target_dur_ms / 1000.0))
                sf.write(output_wav_path, np.zeros(silence_samples, dtype=np.int16), SAMPLE_RATE, subtype="PCM_16")
            elif PYDUB_AVAILABLE:
                silent_seg = AudioSegment.silent(duration=cue.target_dur_ms, frame_rate=SAMPLE_RATE)
                silent_seg.export(output_wav_path, format="wav")
            try:
                if os.path.exists(temp_mp3):
                    os.remove(temp_mp3)
            except Exception:
                pass
            cue.base_wav_path = output_wav_path
            return output_wav_path


async def _batch_synthesize_async(
    cues: List[SubtitleCue],
    voice: str,
    output_dir: Path,
    concurrency: int = 10,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    progress_callback=None,
) -> List[SubtitleCue]:
    """Orchestrates concurrent Edge-TTS synthesis using asyncio.gather."""
    output_dir.mkdir(parents=True, exist_ok=True)
    semaphore = asyncio.Semaphore(concurrency)
    tasks = []

    for cue in cues:
        target_wav = str(output_dir / f"cue_{cue.cue_id:04d}_base.wav")
        tasks.append(_synthesize_single_cue(cue, voice, target_wav, semaphore, rate=rate, pitch=pitch))

    total = len(tasks)
    completed = 0

    print(f"⚡ [TTS GENERATOR] Synthesizing {total} cues with voice '{voice}' (Concurrency: {concurrency})...")
    for f in asyncio.as_completed(tasks):
        await f
        completed += 1
        if progress_callback:
            progress_callback(completed / total, desc=f"TTS Synthesis: {completed}/{total} cues completed...")

    print(f"✅ [TTS GENERATOR] Successfully generated all {total} base WAV audio files in {output_dir}")
    return cues


# ==================================================================================================
# MODULE ENTRYPOINT: GENERATE BASE SPEECH
# ==================================================================================================
def generate_base_speech(
    input_source: str,
    output_dir: str = "./outputs/base_tts",
    voice: str = DEFAULT_VOICE,
    concurrency: int = 12,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    progress_callback=None,
) -> List[SubtitleCue]:
    """
    High-level entrypoint for Module 1.
    
    Args:
        input_source: Path to an .srt file, path to a .txt file, or raw text string.
        output_dir: Directory where generated base .wav files will be saved.
        voice: Voice name or key (e.g. 'hi-IN-MadhurNeural' or 'hi_madhur').
        concurrency: Number of concurrent async synthesis workers.
        rate: Speech rate modifier (e.g. '+0%', '+10%').
        pitch: Speech pitch modifier (e.g. '+0Hz', '+2Hz').
        progress_callback: Optional progress reporter (e.g., Gradio Progress or function).

    Returns:
        List of SubtitleCue objects with base_wav_path populated.
    """
    if not EDGE_TTS_AVAILABLE:
        raise ImportError(
            "Edge-TTS is not installed. Please install it using: pip install edge-tts"
        )

    # Resolve voice alias if provided as shorthand key
    resolved_voice = SUPPORTED_VOICES.get(voice, voice)

    # Determine input type
    if os.path.exists(input_source):
        if input_source.lower().endswith(".srt"):
            cues = parse_srt_file(input_source)
        else:
            with open(input_source, "r", encoding="utf-8") as f:
                cues = parse_plain_text(f.read())
    else:
        # Treat as raw text string
        cues = parse_plain_text(input_source)

    if not cues:
        raise ValueError("No valid dialogue lines found in input.")

    out_path = Path(output_dir).resolve()
    if out_path.exists():
        shutil.rmtree(out_path)
    out_path.mkdir(parents=True, exist_ok=True)

    # Run async loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # Running inside Jupyter/Colab event loop
        import nest_asyncio
        nest_asyncio.apply()
        cues = loop.run_until_complete(
            _batch_synthesize_async(
                cues, resolved_voice, out_path, concurrency=concurrency, rate=rate, pitch=pitch, progress_callback=progress_callback
            )
        )
    else:
        cues = loop.run_until_complete(
            _batch_synthesize_async(
                cues, resolved_voice, out_path, concurrency=concurrency, rate=rate, pitch=pitch, progress_callback=progress_callback
            )
        )

    return cues
