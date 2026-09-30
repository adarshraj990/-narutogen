"""
====================================================================================================
🍥 NARUTOGEN: STREAMLINED KOKORO-82M + RVC DUBBING PIPELINE
====================================================================================================
Architecture: Two-Step High-Fidelity Local Voice Generation & Conversion
1. Step 1 (Local Neural TTS): Kokoro-82M generates expressive, high-quality base audio (.wav) locally.
   - 100% Free, NO API keys, NO cloud subscriptions, runs entirely local / offline.
   - Multi-Language Support: Hindi, Spanish, French, and Portuguese.
2. Step 2 (Voice Conversion): RVC V2 (Retrieval-based Voice Conversion) with RMVPE pitch extraction
   transforms base audio into the desired character or creator voice (e.g., CarryMinati / Naruto).
3. Step 3 (Time-Sync & Assembly): Phase vocoder time-stretching matches dialogue to exact SRT timestamps,
   slicing in 15-minute windows for zero RAM bloat.
4. Minimalist Web UI: Clean Gradio interface with only SRT upload, duration, language dropdown, and dub button.
====================================================================================================
"""

import os
import gc
import re
import sys
import time
import math
import shutil
import tempfile
import zipfile
import urllib.request
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import soundfile as sf
from pydub import AudioSegment
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    gr = None  # type: ignore
    GRADIO_AVAILABLE = False

# Try importing librosa with graceful fallback
try:
    import librosa  # type: ignore
    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None  # type: ignore
    LIBROSA_AVAILABLE = False

# Try importing pysrt with graceful fallback
try:
    import pysrt  # type: ignore
    PYSRT_AVAILABLE = True
except ImportError:
    pysrt = None  # type: ignore
    PYSRT_AVAILABLE = False

# ==================================================================================================
# 1. KOKORO-82M LOCAL TTS ENGINE (REPLACES EDGE-TTS COMPLETELY)
# ==================================================================================================
KOKORO_AVAILABLE = False
KPipeline = None  # type: ignore

try:
    from kokoro import KPipeline as _KP  # type: ignore
    KPipeline = _KP
    KOKORO_AVAILABLE = True
except ImportError:
    KPipeline = None
    KOKORO_AVAILABLE = False

# ==================================================================================================
# 2. PATCH TENSORBOARD SHIM FOR FAIRSEQ & RVC
# ==================================================================================================
def _patch_tensorboard_for_fairseq():
    """Prevents TensorBoard compatibility crashes on Colab when FairSeq is imported."""
    from unittest.mock import MagicMock

    try:
        import tensorboard.compat
        if not hasattr(tensorboard.compat, "notf"):
            tensorboard.compat.notf = MagicMock()
    except Exception:
        pass

    for mod in [
        "torch.utils.tensorboard",
        "torch.utils.tensorboard.writer",
        "torch.utils.tensorboard._embedding",
        "tensorboard",
        "tensorboard.compat",
        "tensorboard.compat.tf",
        "tensorboard.lazy",
    ]:
        if mod not in sys.modules:
            mock_mod = MagicMock()
            mock_mod.SummaryWriter = MagicMock
            mock_mod.FileWriter = MagicMock
            sys.modules[mod] = mock_mod

_patch_tensorboard_for_fairseq()

# In PyTorch 2.6+, torch.load defaults to weights_only=True which breaks RVC checkpoints
try:
    import torch  # type: ignore
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except Exception:
    pass

# Try importing RVC
RVCInference: Any = None
RVC_AVAILABLE = False
try:
    from rvc_python.infer import RVCInference as _RVCClass  # type: ignore
    RVCInference = _RVCClass
    RVC_AVAILABLE = True
except Exception:
    RVCInference = None
    RVC_AVAILABLE = False

# Optional system resource monitoring
try:
    import psutil  # type: ignore
except ImportError:
    psutil = None  # type: ignore


# ==================================================================================================
# 3. GLOBAL CONFIGURATION & MODEL PLACEHOLDER
# ==================================================================================================
# 📌 RVC MODEL DOWNLOAD URL PLACEHOLDER:
# You can paste any direct Hugging Face download link (.zip containing .pth and .index) below:
CONFIGURED_RVC_MODEL_URL = (
    "https://huggingface.co/ivaan2003/ai-rvc/resolve/main/CarryMinati%20-%20Ajey%20Nagar%20-%20Weights.gg%20Model.zip"
)

SAMPLE_RATE = 44100          # High-fidelity sample rate for RVC output
KOKORO_SAMPLE_RATE = 24000   # Native sample rate for Kokoro-82M
FLUSH_INTERVAL_MINUTES = 15  # 15-minute slice window for memory safety
FLUSH_INTERVAL_MS = FLUSH_INTERVAL_MINUTES * 60 * 1000  # 900,000 ms

# Target Dubbing Languages (Kokoro-82M Neural Base Voices)
TARGET_LANGUAGES = {
    "Hindi (Kokoro TTS)": {"lang_code": "h", "voice": "hm_omega", "name": "Hindi Male (Omega)"},
    "Spanish": {"lang_code": "e", "voice": "em_alex", "name": "Spanish Male (Alex)"},
    "French": {"lang_code": "f", "voice": "ff_siwis", "name": "French Female (Siwis)"},
    "Portuguese": {"lang_code": "p", "voice": "pm_alex", "name": "Portuguese Male (Alex)"},
}

# In-memory pipeline cache
_KOKORO_PIPELINES: Dict[str, Any] = {}


def get_memory_stats() -> str:
    """Returns formatted system RAM and GPU VRAM usage."""
    ram_str = "RAM: N/A"
    if psutil:
        mem = psutil.virtual_memory()
        ram_str = f"RAM: {mem.used / (1024**3):.1f}/{mem.total / (1024**3):.1f}GB ({mem.percent}%)"

    vram_str = ""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**3)
            res = torch.cuda.memory_reserved() / (1024**3)
            vram_str = f" | VRAM: {alloc:.2f}GB alloc ({res:.2f}GB res)"
    except Exception:
        pass
    return f"{ram_str}{vram_str}"


# ==================================================================================================
# 3.5 RVC MODEL AUTO-DOWNLOADER & EXTRACTOR (100% SELF-CONTAINED)
# ==================================================================================================
def normalize_huggingface_url(url: str) -> str:
    """Normalizes Hugging Face URLs by converting /blob/ to /resolve/ for direct downloading."""
    url = url.strip()
    if "huggingface.co" in url and "/blob/" in url:
        url = url.replace("/blob/", "/resolve/")
    return url


def _format_bytes(bytes_count: int) -> str:
    """Format bytes to human-readable string (KB, MB, GB)."""
    if bytes_count < 1024:
        return f"{bytes_count} B"
    elif bytes_count < 1024 * 1024:
        return f"{bytes_count / 1024:.1f} KB"
    elif bytes_count < 1024 * 1024 * 1024:
        return f"{bytes_count / (1024 * 1024):.2f} MB"
    else:
        return f"{bytes_count / (1024 * 1024 * 1024):.2f} GB"


def download_file_with_progress(url: str, output_path: Path, chunk_size: int = 1024 * 64) -> Path:
    """Downloads a remote file with a live console progress bar and download stats."""
    url = normalize_huggingface_url(url)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_download_path = output_path.with_suffix(output_path.suffix + ".downloading")

    print(f"\n🌐 [DOWNLOAD] Fetching RVC model archive from:")
    print(f"   🔗 URL: {url}")
    print(f"   📁 Destination: {output_path.name}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }
    req = urllib.request.Request(url, headers=headers)
    start_time = time.time()
    try:
        with urllib.request.urlopen(req, timeout=90) as response, open(temp_download_path, "wb") as out_file:
            content_length = response.headers.get("Content-Length")
            total_size = int(content_length) if content_length and content_length.isdigit() else 0
            downloaded = 0
            last_print = 0

            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                out_file.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_print >= 0.15 or (total_size and downloaded >= total_size):
                    elapsed = max(0.001, now - start_time)
                    speed = downloaded / elapsed
                    speed_str = f"{_format_bytes(int(speed))}/s"

                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        bar_len = 30
                        filled = int(bar_len * downloaded / total_size)
                        bar = "█" * filled + "░" * (bar_len - filled)
                        sys.stdout.write(
                            f"\r   ⏳ [{bar}] {pct:5.1f}% | {_format_bytes(downloaded)} / {_format_bytes(total_size)} | {speed_str} "
                        )
                    else:
                        sys.stdout.write(f"\r   ⏳ Downloaded: {_format_bytes(downloaded)} | {speed_str} ")
                    sys.stdout.flush()
                    last_print = now

            sys.stdout.write("\n")

        if output_path.exists():
            output_path.unlink()
        temp_download_path.rename(output_path)
        total_time = max(0.001, time.time() - start_time)
        print(f"✅ [DOWNLOAD COMPLETE] Saved {output_path.name} ({_format_bytes(output_path.stat().st_size)}) in {total_time:.1f}s.")
        return output_path

    except Exception as e:
        if temp_download_path.exists():
            temp_download_path.unlink()
        print(f"❌ [DOWNLOAD FAILED] Error downloading {url}: {e}")
        raise


def extract_rvc_zip(zip_path: Path, extract_to: Path) -> Tuple[Optional[Path], Optional[Path]]:
    """Extracts zip archive and discovers the largest .pth and .index files."""
    extract_to = Path(extract_to).resolve()
    extract_to.mkdir(parents=True, exist_ok=True)

    print(f"\n📦 [EXTRACT] Unpacking archive: {zip_path.name}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_to)

    pth_file: Optional[Path] = None
    index_file: Optional[Path] = None

    for p in extract_to.rglob("*.pth"):
        if not pth_file or p.stat().st_size > pth_file.stat().st_size:
            pth_file = p

    for p in extract_to.rglob("*.index"):
        if not index_file or p.stat().st_size > index_file.stat().st_size:
            index_file = p

    if pth_file:
        print(f"   🎯 Found Model Weights: {pth_file.name} ({_format_bytes(pth_file.stat().st_size)})")
    if index_file:
        print(f"   🎯 Found Feature Index: {index_file.name} ({_format_bytes(index_file.stat().st_size)})")

    return pth_file, index_file


def setup_model_directories(
    discovered_pth: Path,
    discovered_index: Optional[Path],
    models_dir: Path,
    weights_dir: Path,
    logs_dir: Path,
) -> Dict[str, str]:
    """Organizes model and index files into project paths and standard RVC locations."""
    models_dir = Path(models_dir).resolve()
    weights_dir = Path(weights_dir).resolve()
    logs_dir = Path(logs_dir).resolve()

    models_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    primary_pth = models_dir / "character.pth"
    if discovered_pth.resolve() != primary_pth.resolve():
        shutil.copy2(discovered_pth, primary_pth)

    orig_pth_dest = models_dir / discovered_pth.name
    if orig_pth_dest.resolve() != discovered_pth.resolve() and orig_pth_dest.resolve() != primary_pth.resolve():
        shutil.copy2(discovered_pth, orig_pth_dest)

    rvc_weights_pth = weights_dir / discovered_pth.name
    shutil.copy2(discovered_pth, rvc_weights_pth)
    rvc_canonical_pth = weights_dir / "character.pth"
    if rvc_canonical_pth.resolve() != rvc_weights_pth.resolve():
        shutil.copy2(discovered_pth, rvc_canonical_pth)

    primary_index: Optional[Path] = None
    if discovered_index and discovered_index.exists():
        primary_index = models_dir / "character.index"
        if discovered_index.resolve() != primary_index.resolve():
            shutil.copy2(discovered_index, primary_index)

        orig_index_dest = models_dir / discovered_index.name
        if orig_index_dest.resolve() != discovered_index.resolve() and orig_index_dest.resolve() != primary_index.resolve():
            shutil.copy2(discovered_index, orig_index_dest)

        rvc_logs_index = logs_dir / discovered_index.name
        shutil.copy2(discovered_index, rvc_logs_index)
        rvc_canonical_index = logs_dir / "character.index"
        if rvc_canonical_index.resolve() != rvc_logs_index.resolve():
            shutil.copy2(discovered_index, rvc_canonical_index)

    return {
        "model_path": str(primary_pth),
        "index_path": str(primary_index) if primary_index else "",
        "weights_path": str(rvc_weights_pth),
        "logs_path": str(logs_dir),
    }


def ensure_character_model(
    url: str = CONFIGURED_RVC_MODEL_URL,
    project_root: Optional[str] = None,
    force: bool = False,
) -> Dict[str, str]:
    """Ensures character model exists locally; downloads and extracts if missing."""
    url = normalize_huggingface_url(url)
    root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
    models_dir = root / "models" / "character"
    weights_dir = root / "weights"
    logs_dir = root / "logs" / "character"

    primary_pth = models_dir / "character.pth"
    primary_index = models_dir / "character.index"

    if not force:
        pth_candidates = [
            primary_pth,
            root / "models" / "naruto" / "naruto.pth",
            root / "models" / "naruto" / "naruto-uzumaki-by-mboisuper.pth",
            weights_dir / "character.pth",
        ]
        if weights_dir.exists():
            pth_candidates.extend(list(weights_dir.glob("*.pth")))
        if (root / "models").exists():
            pth_candidates.extend(list((root / "models").rglob("*.pth")))

        for cand in pth_candidates:
            if cand.exists() and cand.stat().st_size > 10_000_000:
                idx_cand = cand.with_suffix(".index")
                found_index = str(idx_cand.resolve()) if idx_cand.exists() else ""
                if not found_index and primary_index.exists():
                    found_index = str(primary_index.resolve())
                return {
                    "model_path": str(cand.resolve()),
                    "index_path": found_index,
                    "weights_path": str(weights_dir),
                    "logs_path": str(logs_dir),
                }

    temp_dir = root / "models" / "temp_download"
    temp_dir.mkdir(parents=True, exist_ok=True)
    zip_dest = temp_dir / "rvc_model.zip"

    download_file_with_progress(url, zip_dest)
    disc_pth, disc_index = extract_rvc_zip(zip_dest, temp_dir / "extracted")

    if not disc_pth or not disc_pth.exists():
        raise RuntimeError(f"No valid .pth weights found in downloaded archive: {url}")

    paths = setup_model_directories(disc_pth, disc_index, models_dir, weights_dir, logs_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)
    return paths


# ==================================================================================================
# 4. SRT SUBTITLE PARSER
# ==================================================================================================
class SubtitleCue:
    """Data representation of a single subtitle dialogue."""
    def __init__(self, cue_id: int, start_ms: int, end_ms: int, text: str):
        self.cue_id = cue_id
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.target_dur_ms = max(200, end_ms - start_ms)
        self.text = text
        self.base_wav_path: Optional[str] = None
        self.rvc_wav_path: Optional[str] = None

    def __repr__(self):
        return f"<Cue #{self.cue_id} [{self.start_ms}ms -> {self.end_ms}ms ({self.target_dur_ms}ms)]: '{self.text[:20]}...'>"


def clean_subtitle_text(text: str) -> str:
    """Cleans dialogue text by stripping HTML tags, subtitle formatting, and sound effect annotations."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\{.*?\}", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_srt_file(srt_path: str, total_video_ms: int) -> Tuple[List[SubtitleCue], int]:
    """Parses SRT subtitles, cleans text, and auto-expands canvas if needed."""
    print(f"📄 [SRT] Parsing subtitles from: {srt_path}")
    cues = []

    if PYSRT_AVAILABLE and pysrt:
        try:
            subs = pysrt.open(srt_path, encoding="utf-8")
            for i, sub in enumerate(subs, start=1):
                clean_txt = clean_subtitle_text(sub.text)
                if not clean_txt:
                    continue
                start_ms = int(sub.start.ordinal)
                end_ms = int(sub.end.ordinal)
                if end_ms <= start_ms:
                    end_ms = start_ms + 1000
                cues.append(SubtitleCue(i, start_ms, end_ms, clean_txt))
        except Exception:
            cues = []

    if not cues:
        # Robust pure-Python regex parser (handles all line endings)
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
                clean_txt = clean_subtitle_text(match.group(4))
                if clean_txt:
                    if end_ms <= start_ms:
                        end_ms = start_ms + 1000
                    cues.append(SubtitleCue(idx, start_ms, end_ms, clean_txt))

    print(f"✅ [SRT] Successfully loaded {len(cues)} valid dialogue cues.")

    # Expand canvas if subtitles exceed video duration
    adjusted_video_ms = total_video_ms
    if cues and cues[-1].end_ms > total_video_ms:
        overhang_sec = (cues[-1].end_ms - total_video_ms) / 1000.0
        print(f"⚠️ [CANVAS] Expanding canvas by {overhang_sec:.1f}s to guarantee dialogue is never cut off.")
        adjusted_video_ms = cues[-1].end_ms + 2000

    return cues, adjusted_video_ms


# ==================================================================================================
# 5. STEP 1: KOKORO-82M LOCAL BATCH SYNTHESIZER
# ==================================================================================================
def get_kokoro_pipeline(lang_code: str = "h") -> Any:
    """Returns or loads a cached KPipeline instance."""
    global _KOKORO_PIPELINES, KPipeline, KOKORO_AVAILABLE

    if not KOKORO_AVAILABLE or KPipeline is None:
        try:
            from kokoro import KPipeline as _KP  # type: ignore
            KPipeline = _KP
            KOKORO_AVAILABLE = True
        except ImportError:
            print("⚡ [KOKORO] 'kokoro' library not found. Auto-installing now...")
            import subprocess
            try:
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-q", "kokoro>=0.8.4", "soundfile"],
                    check=True,
                )
                from kokoro import KPipeline as _KP  # type: ignore
                KPipeline = _KP
                KOKORO_AVAILABLE = True
                print("✅ [KOKORO] Kokoro installed successfully!")
            except Exception as e:
                raise RuntimeError(
                    f"Kokoro-82M TTS is not installed ({e}).\n"
                    "Please run in Colab: !pip install kokoro soundfile"
                )

    lang_code = lang_code.lower()
    if lang_code not in _KOKORO_PIPELINES:
        print(f"📦 [KOKORO TTS] Initializing Kokoro-82M pipeline for language '{lang_code}'...")
        _KOKORO_PIPELINES[lang_code] = KPipeline(lang_code=lang_code)
        print(f"✅ [KOKORO TTS] Kokoro pipeline ready for language '{lang_code}'.")

    return _KOKORO_PIPELINES[lang_code]


def synthesize_single_cue_kokoro(
    cue: SubtitleCue,
    pipeline: Any,
    voice: str,
    output_wav_path: Path,
    speed: float = 1.0,
) -> Path:
    """Generates audio for a single cue using Kokoro-82M and writes a 24kHz WAV file."""
    audio_segments = []
    try:
        for _, _, audio in pipeline(cue.text, voice=voice, speed=speed):
            if audio is not None and len(audio) > 0:
                if hasattr(audio, "cpu"):
                    audio = audio.cpu().numpy()
                audio_segments.append(audio)

        if audio_segments:
            merged = np.concatenate(audio_segments)
        else:
            silence_samples = int(KOKORO_SAMPLE_RATE * (cue.target_dur_ms / 1000.0))
            merged = np.zeros(max(2400, silence_samples), dtype=np.float32)

        sf.write(str(output_wav_path), merged, KOKORO_SAMPLE_RATE)
        cue.base_wav_path = str(output_wav_path)
        return output_wav_path

    except Exception as e:
        print(f"⚠️ [KOKORO SYNTHESIS NOTICE] Cue #{cue.cue_id}: {e}")
        silence_samples = int(KOKORO_SAMPLE_RATE * (cue.target_dur_ms / 1000.0))
        merged = np.zeros(max(2400, silence_samples), dtype=np.float32)
        sf.write(str(output_wav_path), merged, KOKORO_SAMPLE_RATE)
        cue.base_wav_path = str(output_wav_path)
        return output_wav_path


def batch_generate_kokoro_tts(
    cues: List[SubtitleCue],
    lang_code: str = "h",
    voice: str = "hm_omega",
    tts_output_dir: Path = Path("./temp_kokoro"),
    progress_callback=None,
) -> List[Tuple[SubtitleCue, Path]]:
    """Synthesizes all dialogue cues locally using Kokoro-82M."""
    print(f"\n⚡ [STEP 1: KOKORO TTS] Starting local neural synthesis for {len(cues)} cues (Lang: '{lang_code}', Voice: '{voice}')...")
    tts_output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = get_kokoro_pipeline(lang_code)
    cue_file_pairs = []
    total = len(cues)
    start_time = time.time()

    for i, cue in enumerate(cues, start=1):
        out_file = tts_output_dir / f"cue_{cue.cue_id:04d}_kokoro.wav"
        synthesize_single_cue_kokoro(cue, pipeline, voice=voice, output_wav_path=out_file)
        cue_file_pairs.append((cue, out_file))

        if progress_callback and total > 0:
            frac = 0.05 + (i / total) * 0.25  # 5% to 30%
            progress_callback(frac, desc=f"⚡ [Step 1/3] Kokoro TTS: {i}/{total} cues...")

        if i % 25 == 0 or i == total:
            elapsed = time.time() - start_time
            rate = i / max(0.001, elapsed)
            print(f"   ↳ [Kokoro Progress] {i}/{total} cues ({rate:.1f} cues/sec)")

    total_time = time.time() - start_time
    print(f"✅ [STEP 1: KOKORO TTS] Completed {len(cues)} cues in {total_time:.2f}s! {get_memory_stats()}")
    return cue_file_pairs


# ==================================================================================================
# 6. STEP 2: RVC VOICE CONVERSION WITH RMVPE PITCH EXTRACTION
# ==================================================================================================
class CharacterVoiceConverter:
    """Handles RVC voice conversion using pre-trained .pth weights and .index files."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        index_path: Optional[str] = None,
        pitch_shift: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        protect: float = 0.33,
        filter_radius: int = 3,
        resample_sr: int = 0,
        model_url: str = CONFIGURED_RVC_MODEL_URL,
    ):
        self.model_path = model_path
        self.index_path = index_path
        self.pitch_shift = pitch_shift
        self.f0_method = f0_method
        self.index_rate = index_rate
        self.protect = protect
        self.filter_radius = filter_radius
        self.resample_sr = resample_sr
        self.model_url = model_url
        self.engine = None

        # Auto-detect character model or download from model_url
        if not self.model_path or not os.path.exists(self.model_path):
            candidates = [
                Path("models/character/character.pth"),
                Path("weights/character.pth"),
                Path("models/naruto/naruto.pth"),
                Path("weights/naruto.pth"),
            ]
            candidates.extend(list(Path("weights").glob("*.pth")) if Path("weights").exists() else [])
            candidates.extend(list(Path("models").rglob("*.pth")) if Path("models").exists() else [])

            for cand in candidates:
                if cand.exists() and cand.stat().st_size > 10_000_000:
                    self.model_path = str(cand.resolve())
                    if not self.index_path:
                        idx_cand = cand.with_suffix(".index")
                        if idx_cand.exists():
                            self.index_path = str(idx_cand.resolve())
                    break

            # If still not found, automatically download the configured RVC model
            if not self.model_path or not os.path.exists(self.model_path):
                try:
                    print(f"🎙️ [RVC SETUP] Model weights not found locally. Auto-downloading from configured URL...")
                    m_info = ensure_character_model(url=self.model_url)
                    self.model_path = m_info.get("model_path")
                    self.index_path = m_info.get("index_path")
                except Exception as e:
                    print(f"💡 [RVC AUTO-SETUP] Could not download RVC model: {e}")

        if self.model_path and os.path.exists(self.model_path):
            self._init_rvc_engine()
        else:
            print("💡 [RVC] Running in High-Speed Base Kokoro TTS Mode.")

    def _init_rvc_engine(self):
        """Initializes the RVC engine with GPU support."""
        global RVCInference, RVC_AVAILABLE
        print(f"\n🎙️ [STEP 2: RVC] Initializing RVC Engine with model: {Path(self.model_path).name}...")
        if not RVC_AVAILABLE or RVCInference is None:
            try:
                _patch_tensorboard_for_fairseq()
                from rvc_python.infer import RVCInference as _RVCClass  # type: ignore
                RVCInference = _RVCClass
                RVC_AVAILABLE = True
            except Exception as e:
                print(f"💡 [RVC NOTICE] 'rvc-python' not available ({e}). Falling back to base Kokoro TTS.")
                self.engine = None
                return

        if RVCInference is None or not callable(RVCInference):
            print("⚠️ [RVC LOAD ERROR] RVCInference is unavailable. Falling back to baseline TTS.")
            self.engine = None
            return

        try:
            device = "cuda:0"
            try:
                import torch  # type: ignore
                if not torch.cuda.is_available():
                    device = "cpu"
                    print("⚠️ [RVC NOTICE] CUDA unavailable, running RVC on CPU.")
            except Exception:
                device = "cpu"

            self.engine = RVCInference(device=device)  # type: ignore
            self.engine.load_model(self.model_path, index_path=self.index_path or "", version="v2")
            if hasattr(self.engine, "set_params"):
                self.engine.set_params(
                    f0method=self.f0_method,
                    f0up_key=self.pitch_shift,
                    index_rate=self.index_rate,
                    protect=self.protect,
                    filter_radius=getattr(self, "filter_radius", 3),
                    resample_sr=getattr(self, "resample_sr", 0),
                )
            print(f"✅ [STEP 2: RVC] RVC Model Loaded on {device} with RMVPE pitch extraction! {get_memory_stats()}")
        except Exception as e:
            print(f"⚠️ [RVC LOAD ERROR] Could not load RVC engine: {e}. Falling back to baseline TTS.")
            self.engine = None

    def convert_file(self, input_wav: Path, output_wav: Path) -> bool:
        """Converts a single audio file to target character voice."""
        if not self.engine:
            shutil.copyfile(input_wav, output_wav)
            return True

        try:
            if hasattr(self.engine, "set_params"):
                self.engine.set_params(
                    f0method=self.f0_method,
                    f0up_key=self.pitch_shift,
                    index_rate=self.index_rate,
                    protect=self.protect,
                )
            try:
                self.engine.infer_file(str(input_wav), str(output_wav))
            except TypeError:
                self.engine.infer_file(
                    input_path=str(input_wav),
                    output_path=str(output_wav),
                    pitch=self.pitch_shift,
                    f0method=self.f0_method,
                )
            return True
        except Exception as e:
            print(f"⚠️ [RVC CONVERT ERROR] {input_wav.name}: {e}. Retaining baseline audio.")
            shutil.copyfile(input_wav, output_wav)
            return False

    def batch_convert(
        self,
        cue_file_pairs: List[Tuple[SubtitleCue, Path]],
        rvc_output_dir: Path,
        progress_callback=None,
    ) -> List[Tuple[SubtitleCue, Path]]:
        """Batch-converts all dialogue snippets to the target voice."""
        rvc_output_dir.mkdir(parents=True, exist_ok=True)
        total = len(cue_file_pairs)

        if not self.engine:
            print("⚡ [STEP 2: RVC] Skipping RVC (Base Kokoro TTS mode active).")
            return cue_file_pairs

        print(f"\n🍥 [STEP 2: RVC] Batch-converting {total} cues to character voice (Pitch: {self.pitch_shift}, RMVPE)...")
        results = []
        start_time = time.time()

        for idx, (cue, base_wav) in enumerate(cue_file_pairs, start=1):
            out_rvc_wav = rvc_output_dir / f"cue_{cue.cue_id:04d}_rvc.wav"
            self.convert_file(base_wav, out_rvc_wav)
            cue.rvc_wav_path = str(out_rvc_wav)
            results.append((cue, out_rvc_wav))

            # Periodic GPU VRAM clearing to prevent OOM
            if idx % 20 == 0:
                try:
                    import torch  # type: ignore
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                gc.collect()

            if progress_callback and total > 0:
                frac = 0.30 + (idx / total) * 0.50  # 30% to 80%
                progress_callback(frac, desc=f"🍥 [Step 2/3] RVC Converting: {idx}/{total} cues (RMVPE)...")

            if idx % 20 == 0 or idx == total:
                elapsed = time.time() - start_time
                rate = idx / max(0.001, elapsed)
                print(f"   ↳ [RVC Progress] {idx}/{total} cues ({rate:.1f} cues/sec)")

        total_time = time.time() - start_time
        print(f"✅ [STEP 2: RVC] Completed {total} voice conversions in {total_time:.2f}s ({total_time/60:.2f} mins)! {get_memory_stats()}")
        return results


# ==================================================================================================
# 7. STEP 3: TIME-SYNCHRONIZATION & MASTER TIMELINE CANVAS ASSEMBLY
# ==================================================================================================
def time_sync_audio_file(audio_path: Path, target_dur_ms: int) -> AudioSegment:
    """
    Fits audio chunk to the exact subtitle duration:
    - If audio is too long: pitch-preserving time-stretch (librosa phase vocoder).
    - If audio is too short: natural silence padding.
    """
    try:
        y, sr = sf.read(str(audio_path), dtype="float32")
        if len(y.shape) > 1:
            y = y.mean(axis=1)

        actual_dur_ms = int(round((len(y) / sr) * 1000.0))

        if actual_dur_ms > target_dur_ms and LIBROSA_AVAILABLE and librosa:
            rate = float(actual_dur_ms) / float(target_dur_ms)
            rate = min(rate, 2.5)  # Cap compression at 2.5x to preserve intelligibility
            y_stretched = librosa.effects.time_stretch(y, rate=rate)
            stretched_int16 = (np.clip(y_stretched, -1.0, 1.0) * 32767).astype(np.int16)

            seg = AudioSegment(
                data=stretched_int16.tobytes(),
                sample_width=2,
                frame_rate=sr,
                channels=1,
            )
            if len(seg) > target_dur_ms:
                seg = seg[:target_dur_ms]
            elif len(seg) < target_dur_ms:
                seg = seg + AudioSegment.silent(duration=target_dur_ms - len(seg), frame_rate=sr)
            return seg

        elif actual_dur_ms < target_dur_ms:
            audio_int16 = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
            seg = AudioSegment(
                data=audio_int16.tobytes(),
                sample_width=2,
                frame_rate=sr,
                channels=1,
            )
            pad_ms = target_dur_ms - actual_dur_ms
            return seg + AudioSegment.silent(duration=pad_ms, frame_rate=sr)

        else:
            audio_int16 = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
            return AudioSegment(
                data=audio_int16.tobytes(),
                sample_width=2,
                frame_rate=sr,
                channels=1,
            )

    except Exception:
        seg = AudioSegment.from_file(str(audio_path))
        if len(seg) > target_dur_ms:
            return seg[:target_dur_ms]
        elif len(seg) < target_dur_ms:
            return seg + AudioSegment.silent(duration=target_dur_ms - len(seg), frame_rate=SAMPLE_RATE)
        return seg


def assemble_master_audio(
    cue_file_pairs: List[Tuple[SubtitleCue, Path]],
    total_video_ms: int,
    output_wav_path: Path,
    parts_dir: Path,
    progress_callback=None,
) -> Path:
    """Overlays dialogue onto the timeline, flushing 15-min windows to disk for zero-OOM safety."""
    print(f"\n🎧 [STEP 3: SYNC & ASSEMBLY] Assembling master canvas ({total_video_ms/1000/60:.2f} mins)...")
    parts_dir.mkdir(parents=True, exist_ok=True)

    num_windows = math.ceil(total_video_ms / FLUSH_INTERVAL_MS)
    current_window_idx = 0
    intermediate_files: List[Path] = []

    def init_window_canvas(win_idx: int) -> AudioSegment:
        win_start = win_idx * FLUSH_INTERVAL_MS
        win_dur = min(FLUSH_INTERVAL_MS, total_video_ms - win_start)
        return AudioSegment.silent(duration=win_dur, frame_rate=SAMPLE_RATE)

    active_canvas = init_window_canvas(current_window_idx)
    overflow_buffer: List[Tuple[int, AudioSegment]] = []
    total_cues = len(cue_file_pairs)

    for i, (cue, audio_file) in enumerate(cue_file_pairs, start=1):
        synced_seg = time_sync_audio_file(audio_file, cue.target_dur_ms)
        cue_window_idx = cue.start_ms // FLUSH_INTERVAL_MS

        while cue_window_idx > current_window_idx:
            part_num = current_window_idx + 1
            part_path = parts_dir / f"part_{part_num:03d}.wav"
            print(f"💾 [FLUSH] Exporting 15-min Part {part_num} to disk...")
            active_canvas.export(str(part_path), format="wav")
            intermediate_files.append(part_path)

            del active_canvas
            gc.collect()

            current_window_idx += 1
            active_canvas = init_window_canvas(current_window_idx)

            for ov_pos, ov_seg in overflow_buffer:
                active_canvas = active_canvas.overlay(ov_seg, position=ov_pos)
            overflow_buffer.clear()

        win_start_ms = current_window_idx * FLUSH_INTERVAL_MS
        local_pos_ms = cue.start_ms - win_start_ms
        win_dur_ms = len(active_canvas)

        if local_pos_ms + len(synced_seg) > win_dur_ms:
            split_point = win_dur_ms - local_pos_ms
            current_slice = synced_seg[:split_point]
            overflow_slice = synced_seg[split_point:]

            active_canvas = active_canvas.overlay(current_slice, position=local_pos_ms)
            overflow_buffer.append((0, overflow_slice))
        else:
            active_canvas = active_canvas.overlay(synced_seg, position=local_pos_ms)

        if progress_callback and total_cues > 0:
            frac = 0.80 + (i / total_cues) * 0.15  # 80% to 95%
            progress_callback(frac, desc=f"🎧 [Step 3/3] Synchronizing dialogue {i}/{total_cues}...")

    # Flush final window
    part_num = current_window_idx + 1
    part_path = parts_dir / f"part_{part_num:03d}.wav"
    active_canvas.export(str(part_path), format="wav")
    intermediate_files.append(part_path)
    del active_canvas
    gc.collect()

    # Outro preservation: Pad trailing silence if video is longer than subtitles
    current_window_idx += 1
    while current_window_idx < num_windows:
        part_num = current_window_idx + 1
        part_path = parts_dir / f"part_{part_num:03d}.wav"
        trailing_canvas = init_window_canvas(current_window_idx)
        trailing_canvas.export(str(part_path), format="wav")
        intermediate_files.append(part_path)
        del trailing_canvas
        gc.collect()
        current_window_idx += 1

    # Concatenate all parts via FFmpeg (0 MB RAM overhead)
    concat_txt = parts_dir / "concat_list.txt"
    with open(concat_txt, "w", encoding="utf-8") as f:
        for p in intermediate_files:
            f.write(f"file '{p.resolve()}'\n")

    if shutil.which("ffmpeg"):
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_txt),
            "-c", "copy",
            str(output_wav_path),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    else:
        with sf.SoundFile(str(output_wav_path), mode="w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16") as outfile:
            for p in intermediate_files:
                with sf.SoundFile(str(p), mode="r") as infile:
                    while True:
                        data = infile.read(65536, dtype="int16")
                        if len(data) == 0:
                            break
                        outfile.write(data)

    print(f"🎉 [MASTER AUDIO COMPLETE] Saved to: {output_wav_path}")
    return output_wav_path


# ==================================================================================================
# 8. MASTER PIPELINE CONTROLLER
# ==================================================================================================
def run_narutogen_pipeline(
    srt_file: str,
    output_audio: str = "narutogen_dubbed_audio.wav",
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
    target_language: str = "Hindi (Kokoro TTS)",
    rvc_model_path: Optional[str] = None,
    rvc_index_path: Optional[str] = None,
    pitch_shift: int = 0,
    progress_callback=None,
) -> Dict[str, Any]:
    """Automates the entire SRT -> Kokoro-82M -> RVC -> Stitched Master Audio workflow."""
    pipeline_start = time.time()

    # Step 0: Total duration
    total_video_ms = ((hours * 3600) + (minutes * 60) + seconds) * 1000
    if total_video_ms <= 0:
        total_video_ms = 60000

    # Staging folders
    work_dir = Path(tempfile.gettempdir()) / "narutogen_workspace"
    tts_dir = work_dir / "step1_kokoro"
    rvc_dir = work_dir / "step2_rvc"
    parts_dir = work_dir / "step3_parts"

    for d in [tts_dir, rvc_dir, parts_dir]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(0.02, desc="📄 Parsing SRT subtitles...")

    # Step 1: Parse SRT
    cues, adjusted_video_ms = parse_srt_file(srt_file, total_video_ms)
    if not cues:
        raise ValueError("No valid dialogue cues found in the uploaded SRT file.")

    # Resolve Kokoro voice and language
    lang_info = TARGET_LANGUAGES.get(target_language, TARGET_LANGUAGES["Hindi (Kokoro TTS)"])
    lang_code = lang_info["lang_code"]
    voice = lang_info["voice"]

    # Step 2: Kokoro-82M Local Neural TTS
    cue_tts_pairs = batch_generate_kokoro_tts(
        cues=cues,
        lang_code=lang_code,
        voice=voice,
        tts_output_dir=tts_dir,
        progress_callback=progress_callback,
    )

    # Step 3: RVC Voice Conversion
    converter = CharacterVoiceConverter(
        model_path=rvc_model_path,
        index_path=rvc_index_path,
        pitch_shift=pitch_shift,
        model_url=CONFIGURED_RVC_MODEL_URL,
    )
    cue_rvc_pairs = converter.batch_convert(cue_tts_pairs, rvc_output_dir=rvc_dir, progress_callback=progress_callback)

    # Step 4: Time-sync and master canvas assembly
    out_audio_path = Path(output_audio).resolve()
    final_audio = assemble_master_audio(
        cue_file_pairs=cue_rvc_pairs,
        total_video_ms=adjusted_video_ms,
        output_wav_path=out_audio_path,
        parts_dir=parts_dir,
        progress_callback=progress_callback,
    )

    if progress_callback:
        progress_callback(1.0, desc="🏆 Dubbing Pipeline Complete!")

    total_time = time.time() - pipeline_start
    audio_info = sf.info(str(final_audio))

    return {
        "audio_path": str(final_audio),
        "elapsed_seconds": total_time,
        "audio_duration_sec": audio_info.duration,
        "cues_count": len(cues),
        "rvc_active": converter.engine is not None,
    }


# ==================================================================================================
# 9. MINIMAL & CLEAN GRADIO WEB UI INTERFACE
# ==================================================================================================
CUSTOM_CSS = """
.gradio-container {
    max-width: 820px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.naruto-header {
    text-align: center;
    padding: 24px 20px;
    background: linear-gradient(135deg, #181824 0%, #101018 100%);
    border-radius: 16px;
    border: 1px solid rgba(255, 140, 0, 0.25);
    margin-bottom: 22px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
}

.naruto-title {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff8c00 0%, #ff4500 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}

.naruto-subtitle {
    font-size: 0.95rem;
    color: #94a3b8;
}

.btn-start {
    background: linear-gradient(135deg, #ff8c00 0%, #ff4500 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
    padding: 14px 24px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 18px rgba(255, 69, 0, 0.35) !important;
    cursor: pointer !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

.btn-start:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(255, 69, 0, 0.55) !important;
}

.status-card {
    padding: 16px;
    border-radius: 12px;
    background: rgba(255, 140, 0, 0.05);
    border: 1px solid rgba(255, 140, 0, 0.2);
    color: #e2e8f0;
}
"""


def gradio_dubbing_handler(
    srt_file_obj,
    hours: float,
    minutes: float,
    seconds: float,
    target_language: str,
    progress=None,
):
    """Clean minimal handler connecting the UI to the pipeline."""
    if progress is None and GRADIO_AVAILABLE and gr is not None:
        try:
            progress = gr.Progress(track_tqdm=True)
        except Exception:
            progress = None
    if srt_file_obj is None:
        return "### ⚠️ Please upload an SRT subtitle file (*.srt) to begin.", None

    srt_path = getattr(srt_file_obj, "name", str(srt_file_obj))

    out_dir = Path(tempfile.gettempdir()) / "narutogen_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    out_audio = str(out_dir / f"master_dubbed_{stamp}.wav")

    try:
        results = run_narutogen_pipeline(
            srt_file=srt_path,
            output_audio=out_audio,
            hours=int(hours or 0),
            minutes=int(minutes or 0),
            seconds=int(seconds or 0),
            target_language=target_language,
            progress_callback=progress,
        )

        elapsed = results["elapsed_seconds"]
        dur = results["audio_duration_sec"]
        rvc_status = "✅ Active (RVC Character Model Loaded)" if results["rvc_active"] else "⚡ Kokoro Neural TTS Baseline"

        status_markdown = f"""
### 🎉 Dubbing Completed Successfully!

| Metric | Result |
| :--- | :--- |
| **Target Language** | {target_language} |
| **Voice Conversion (RVC)** | {rvc_status} |
| **Audio Duration** | `{int(dur//60):02d}:{int(dur%60):02d}` ({dur:.2f}s) |
| **Dialogue Lines Processed** | {results["cues_count"]} cues |
| **Total Processing Time** | **{elapsed:.1f}s** ({elapsed/60:.2f} mins) |

*Listen to or download your master dubbed audio file below.*
"""
        return status_markdown, results["audio_path"]

    except Exception as e:
        err_msg = f"""
### ❌ Dubbing Failed
An error occurred during execution:
```text
{str(e)}
```
*Tip: Ensure your SRT file contains valid timestamps and dialogue.*
"""
        return err_msg, None


def build_ui():
    """Constructs the clean, minimal Gradio interface (strictly 4 requested elements)."""
    if not GRADIO_AVAILABLE or gr is None:
        raise RuntimeError(
            "Gradio is not installed in the environment.\n"
            "Please run: pip install gradio"
        )

    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="AI Voice Dubbing Studio (Kokoro-82M + RVC)") as demo:

        gr.HTML(
            """
            <div class="naruto-header">
                <div class="naruto-title">🎙️ AI Voice Dubbing Studio</div>
                <div class="naruto-subtitle">
                    Kokoro-82M Neural TTS + RVC Voice Conversion • 100% Local & Free
                </div>
            </div>
            """
        )

        with gr.Group():
            # 1. File Upload (Single .srt file)
            srt_input = gr.File(
                label="1. Translated SRT Subtitles (*.srt)",
                file_types=[".srt"],
                file_count="single",
            )

            # 2. Language Selection Dropdown
            lang_dropdown = gr.Dropdown(
                choices=list(TARGET_LANGUAGES.keys()),
                value="Hindi (Kokoro TTS)",
                label="2. Target Dubbing Language",
                info="Generates base speech using local Kokoro-82M neural model",
            )

            # 3. Video Duration Inputs (Hours, Minutes, Seconds)
            gr.Markdown("#### 3. Video Duration (Timeline Canvas Setup)")
            with gr.Row():
                h_input = gr.Number(label="Hours", value=0, precision=0, minimum=0)
                m_input = gr.Number(label="Minutes", value=0, precision=0, minimum=0)
                s_input = gr.Number(label="Seconds", value=48, precision=0, minimum=0)

            # 4. Action Button
            start_btn = gr.Button(
                "⚡ Start Dubbing",
                variant="primary",
                size="lg",
                elem_classes="btn-start",
            )

        # Output Section
        gr.Markdown("### 🎧 Dubbed Master Audio Output")
        status_box = gr.Markdown(
            """
            <div class="status-card">
                <b>Ready to dub.</b> Upload your <code>.srt</code> file, select the target language, and click <b>Start Dubbing</b>.
            </div>
            """
        )
        audio_player = gr.Audio(
            label="Master Dubbed Audio (.wav)",
            type="filepath",
            interactive=False,
        )

        # Connect button event
        start_btn.click(
            fn=gradio_dubbing_handler,
            inputs=[
                srt_input,
                h_input,
                m_input,
                s_input,
                lang_dropdown,
            ],
            outputs=[
                status_box,
                audio_player,
            ],
        )

    return demo


# Alias exports for backwards-compatibility
run_pipeline = run_narutogen_pipeline
DEFAULT_BATCH_SIZE = 10


# ==================================================================================================
# 10. PUBLIC LAUNCH & MODEL CONFIGURATION SPOT
# ==================================================================================================
# ⚙️ USER MODEL CONFIGURATION:
# Paste your custom RVC model download link below (Hugging Face .zip containing .pth and .index files).
# The pipeline will automatically download and extract it to models/character/ and weights/.
CUSTOM_RVC_MODEL_DOWNLOAD_URL = (
    "https://huggingface.co/ivaan2003/ai-rvc/resolve/main/CarryMinati%20-%20Ajey%20Nagar%20-%20Weights.gg%20Model.zip"
)

if __name__ == "__main__":
    demo = build_ui()
    print("\n" + "=" * 70)
    print("🚀 [DUBBING STUDIO] Launching Interactive Web Interface...")
    print(f"🎙️ Configured RVC Model URL: {CUSTOM_RVC_MODEL_DOWNLOAD_URL}")
    print("=" * 70 + "\n")

    try:
        demo.queue().launch(
            share=True,
            server_name="0.0.0.0",
            server_port=7860,
            show_error=True,
        )
    except OSError:
        demo.queue().launch(
            share=True,
            server_name="0.0.0.0",
            show_error=True,
        )
