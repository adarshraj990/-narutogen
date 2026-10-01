"""
====================================================================================================
🍥 PRODUCTION MULTI-LANGUAGE DUBBING PIPELINE (HUGGING FACE SPACES & COLAB OPTIMIZED)
====================================================================================================
Architecture & Core Features:
1. 100% Free & Local TTS: Powered by Kokoro-82M Neural TTS (zero edge-tts, zero API keys).
2. Smart Chunking & Memory Optimization (HF Spaces Safe):
   - Automatically breaks down long scripts/SRT subtitles (even 2+ hour videos) into 5-minute chunks.
   - Processes each chunk sequentially through Kokoro TTS and RVC to guarantee zero OOM crashes.
   - Empties GPU VRAM (`torch.cuda.empty_cache()`) and forces garbage collection after every chunk.
   - Merges processed chunk slices back together via FFmpeg lossless stream copy (with streaming fallback).
3. Sequential Multi-Language Processing:
   - Supports selecting multiple languages in a single run (Hindi, Spanish, French, Portuguese).
   - Processes each language sequentially: finishes Language 1 (all chunks + master merge), then auto-starts Language 2.
4. RVC Voice Conversion (RMVPE):
   - Converts base speech into target character voice with RMVPE pitch extraction.
   - Pre-configured backend placeholder for RVC model (.zip containing .pth and .index).
5. Clean Minimalist Gradio UI:
   - Single SRT file upload
   - Video Duration inputs (Hours, Minutes, Seconds)
   - Language Selection Checkboxes (Multiple Choice)
   - "Start Dubbing" Action button
   - Absolutely NO manual file uploads for .pth or .index models.
====================================================================================================
"""

import os
import gc
import re
import sys

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import time
import math
import shutil
import tempfile
import zipfile
import urllib.request
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import soundfile as sf
from pydub import AudioSegment

# Backward compatibility shim for Gradio 4.x with modern huggingface_hub
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class _HfFolderShim:
            path_token = os.path.expanduser("~/.cache/huggingface/token")

            @classmethod
            def save_token(cls, token: str):
                try:
                    os.makedirs(os.path.dirname(cls.path_token), exist_ok=True)
                    with open(cls.path_token, "w", encoding="utf-8") as f:
                        f.write(token)
                except Exception:
                    pass

            @classmethod
            def get_token(cls):
                token = os.environ.get("HF_TOKEN")
                if token:
                    return token
                if os.path.exists(cls.path_token):
                    try:
                        with open(cls.path_token, "r", encoding="utf-8") as f:
                            return f.read().strip()
                    except Exception:
                        return None
                return None

            @classmethod
            def delete_token(cls):
                if os.path.exists(cls.path_token):
                    try:
                        os.remove(cls.path_token)
                    except OSError:
                        pass

        huggingface_hub.HfFolder = _HfFolderShim
except Exception:
    pass

# Backward compatibility patch for Gradio 4.44.1 with Starlette 1.0+ TemplateResponse (prevents 500 error)
try:
    from starlette.templating import Jinja2Templates
    _orig_template_response = Jinja2Templates.TemplateResponse

    def _safe_template_response(self, *args, **kwargs):
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
            name = args[0]
            context = args[1]
            request = context.get("request")
            if request is not None:
                try:
                    return _orig_template_response(self, request, name, context, *args[2:], **kwargs)
                except TypeError:
                    pass
        return _orig_template_response(self, *args, **kwargs)

    Jinja2Templates.TemplateResponse = _safe_template_response
except Exception:
    pass

# Backward compatibility patch for Gradio 4.44.1 with modern Pydantic boolean schemas
try:
    import gradio_client.utils as _gc_utils
    _orig_js2py = _gc_utils._json_schema_to_python_type

    def _patched_js2py(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_js2py(schema, defs)

    _gc_utils._json_schema_to_python_type = _patched_js2py

    _orig_get_type = _gc_utils.get_type

    def _patched_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)

    _gc_utils.get_type = _patched_get_type
except Exception:
    pass

# Graceful Gradio import
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    gr = None  # type: ignore
    GRADIO_AVAILABLE = False

# Graceful librosa import
try:
    import librosa  # type: ignore
    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None  # type: ignore
    LIBROSA_AVAILABLE = False

# Graceful pysrt import
try:
    import pysrt  # type: ignore
    PYSRT_AVAILABLE = True
except ImportError:
    pysrt = None  # type: ignore
    PYSRT_AVAILABLE = False

# Optional psutil system monitoring
try:
    import psutil  # type: ignore
except ImportError:
    psutil = None  # type: ignore


# ==================================================================================================
# 1. PATCH TENSORBOARD SHIM FOR FAIRSEQ & RVC
# ==================================================================================================
def _patch_tensorboard_for_fairseq():
    """Prevents TensorBoard compatibility crashes on Colab/HF Spaces when FairSeq is imported."""
    from unittest.mock import MagicMock
    try:
        import tensorboard.compat
        if not hasattr(tensorboard.compat, "notf"):
            tensorboard.compat.notf = MagicMock()
    except Exception:
        pass

    for mod in [
        "torch.utils.tensorboard", "torch.utils.tensorboard.writer",
        "torch.utils.tensorboard._embedding", "tensorboard",
        "tensorboard.compat", "tensorboard.compat.tf", "tensorboard.lazy",
    ]:
        if mod not in sys.modules:
            mock_mod = MagicMock()
            mock_mod.SummaryWriter = MagicMock
            mock_mod.FileWriter = MagicMock
            sys.modules[mod] = mock_mod

_patch_tensorboard_for_fairseq()

# PyTorch 2.6+ weights_only compatibility patch for RVC checkpoints
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


# ==================================================================================================
# 2. GLOBAL CONFIGURATION & RVC MODEL PLACEHOLDER
# ==================================================================================================
# 📌 [INSERT RVC MODEL DOWNLOAD LINK HERE]:
# Paste your direct Hugging Face or direct download link (.zip containing .pth and .index) below.
# The pipeline automatically downloads and extracts it into models/character/ and weights/.
CONFIGURED_RVC_MODEL_URL = (
    "https://huggingface.co/ivaan2003/ai-rvc/resolve/main/CarryMinati%20-%20Ajey%20Nagar%20-%20Weights.gg%20Model.zip"
)

# Audio sample rates
SAMPLE_RATE = 44100          # High-fidelity master sample rate
KOKORO_SAMPLE_RATE = 24000   # Native sample rate for Kokoro-82M

# Chunking Configuration (Crucial for HF Spaces 16GB RAM limit & 2-Hour video safety)
CHUNK_DURATION_MINUTES = 5   # 5-minute slices guarantee < 1.5 GB memory footprint
CHUNK_DURATION_MS = CHUNK_DURATION_MINUTES * 60 * 1000  # 300,000 ms

# Supported Languages Catalog (Kokoro-82M Neural Voices)
SUPPORTED_LANGUAGES = {
    "Hindi": {
        "lang_code": "h",
        "voice": "hm_omega",
        "label": "Hindi (Male - Omega)",
        "fallback_voices": ["hm_psi", "hf_alpha"],
    },
    "Spanish": {
        "lang_code": "e",
        "voice": "em_alex",
        "label": "Spanish (Male - Alex)",
        "fallback_voices": ["em_santa", "ef_dora"],
    },
    "French": {
        "lang_code": "f",
        "voice": "ff_siwis",
        "label": "French (Female - Siwis)",
        "fallback_voices": [],
    },
    "Portuguese": {
        "lang_code": "p",
        "voice": "pm_alex",
        "label": "Portuguese (Male - Alex)",
        "fallback_voices": ["pm_santa", "pf_dora"],
    },
}

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
# 3. RVC MODEL AUTO-DOWNLOADER & EXTRACTOR
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
    """Downloads remote file with live console progress reporting."""
    url = normalize_huggingface_url(url)
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_download_path = output_path.with_suffix(output_path.suffix + ".downloading")

    print(f"\n🌐 [DOWNLOAD] Fetching RVC model archive from:")
    print(f"   🔗 URL: {url}")
    print(f"   📁 Destination: {output_path.name}")

    headers = {"User-Agent": "Mozilla/5.0"}
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
                if now - last_print >= 0.2 or (total_size and downloaded >= total_size):
                    elapsed = max(0.001, now - start_time)
                    speed = downloaded / elapsed
                    speed_str = f"{_format_bytes(int(speed))}/s"

                    if total_size > 0:
                        pct = (downloaded / total_size) * 100
                        bar_len = 25
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
            weights_dir / "character.pth",
            root / "models" / "naruto" / "naruto.pth",
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
# 4. SRT SUBTITLE PARSER & TEXT CLEANER
# ==================================================================================================
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
        return f"<Cue #{self.cue_id} [{self.start_ms}ms -> {self.end_ms}ms ({self.target_dur_ms}ms)]: '{self.text[:20]}...'>"


def clean_subtitle_text(text: str) -> str:
    """Cleans dialogue text by stripping HTML tags, subtitle formatting, and sound effect annotations."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\{.*?\}", "", text)
    return re.sub(r"\s+", " ", text).strip()


def parse_timestamp_ms(ts_str: str) -> int:
    """
    Safely converts any subtitle timestamp string into milliseconds.
    Handles:
      - 00:01:23,456
      - 00:01:23.456
      - 0:01:23,45
      - 00:01:23
      - 01:23.456 (minutes:seconds)
    NEVER throws IndexError or ValueError.
    """
    if not ts_str or not isinstance(ts_str, str):
        return 0
    try:
        clean_ts = ts_str.strip().replace(",", ".")
        # Standard format: (hours):(minutes):(seconds).(fraction)
        m = re.match(r"^(\d+):(\d{1,2}):(\d{1,2})(?:\.(\d+))?", clean_ts)
        if m:
            h = int(m.group(1))
            minute = int(m.group(2))
            s = int(m.group(3))
            ms_raw = m.group(4) or "0"
            ms = int(ms_raw.ljust(3, "0")[:3])
            return (h * 3600 + minute * 60 + s) * 1000 + ms

        # Short format without hours: (minutes):(seconds).(fraction)
        m2 = re.match(r"^(\d{1,2}):(\d{1,2})(?:\.(\d+))?", clean_ts)
        if m2:
            minute = int(m2.group(1))
            s = int(m2.group(2))
            ms_raw = m2.group(3) or "0"
            ms = int(ms_raw.ljust(3, "0")[:3])
            return (minute * 60 + s) * 1000 + ms
    except Exception:
        pass
    return 0


def read_text_safely(file_path: str) -> str:
    """Reads a text file with multiple encoding attempts to prevent decode errors."""
    for enc in ["utf-8-sig", "utf-8", "latin-1", "cp1252"]:
        try:
            with open(file_path, "r", encoding=enc, errors="replace") as f:
                return f.read()
        except Exception:
            continue
    try:
        with open(file_path, "rb") as f:
            return f.read().decode("utf-8", errors="ignore")
    except Exception:
        return ""


def parse_srt_file(srt_path: str, total_video_ms: int) -> Tuple[List[SubtitleCue], int]:
    """
    Parses SRT subtitles with a multi-layer, fail-safe engine:
    1. Item-by-item pysrt parsing with try/except protection per cue.
    2. Robust regex scanner handling irregular newlines, malformed blocks, and missing indices.
    3. Line-by-line tolerant fallback scanner for unconventional subtitle files.
    NEVER throws IndexError: list index out of range.
    """
    print(f"📄 [SRT] Parsing subtitles from: {srt_path}")
    cues: List[SubtitleCue] = []

    # Strategy 1: pysrt with per-item exception handling
    if PYSRT_AVAILABLE and pysrt:
        try:
            subs = pysrt.open(srt_path, encoding="utf-8")
            for i, sub in enumerate(subs, start=1):
                try:
                    txt = getattr(sub, "text", "")
                    clean_txt = clean_subtitle_text(txt)
                    if not clean_txt:
                        continue
                    start_val = getattr(sub.start, "ordinal", None)
                    end_val = getattr(sub.end, "ordinal", None)
                    if start_val is None:
                        continue
                    start_ms = int(start_val)
                    end_ms = int(end_val) if end_val is not None else start_ms + 1000
                    if end_ms <= start_ms:
                        end_ms = start_ms + 1000
                    cues.append(SubtitleCue(i, start_ms, end_ms, clean_txt))
                except Exception:
                    continue
        except Exception:
            pass

    # Strategy 2: Robust Universal Regex Scanner (tolerant of malformed blocks & extra newlines)
    if not cues:
        raw_content = read_text_safely(srt_path).replace("\r\n", "\n").replace("\r", "\n")
        ts_pattern = re.compile(
            r"(?:(?<=\n)|^)\s*(?:(\d+)\s*\n)?\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3})[^\n]*\n([\s\S]*?)(?=(?:\n\s*(?:\d+\s*\n)?\s*\d{1,2}:\d{2}:\d{2}[,\.]\d{1,3}\s*-->)|\Z)"
        )
        for i, match in enumerate(ts_pattern.finditer(raw_content), start=1):
            try:
                raw_idx = match.group(1)
                cue_id = int(raw_idx) if raw_idx and raw_idx.isdigit() else i
                start_ms = parse_timestamp_ms(match.group(2))
                end_ms = parse_timestamp_ms(match.group(3))
                text_part = clean_subtitle_text(match.group(4))
                if not text_part:
                    continue
                if end_ms <= start_ms:
                    end_ms = start_ms + 1000
                cues.append(SubtitleCue(cue_id, start_ms, end_ms, text_part))
            except Exception:
                continue

    # Strategy 3: Line-by-line tolerant fallback
    if not cues:
        raw_content = read_text_safely(srt_path).replace("\r\n", "\n").replace("\r", "\n")
        lines = [line.strip() for line in raw_content.split("\n")]
        i = 0
        while i < len(lines):
            try:
                line = lines[i]
                if "-->" in line:
                    arrow_parts = line.split("-->")
                    if len(arrow_parts) >= 2:
                        start_ms = parse_timestamp_ms(arrow_parts[0])
                        end_ms = parse_timestamp_ms(arrow_parts[1])
                        text_lines = []
                        i += 1
                        while i < len(lines) and lines[i] and "-->" not in lines[i]:
                            if lines[i].isdigit() and (i + 1 < len(lines)) and ("-->" in lines[i + 1]):
                                break
                            text_lines.append(lines[i])
                            i += 1
                        clean_txt = clean_subtitle_text(" ".join(text_lines))
                        if clean_txt:
                            if end_ms <= start_ms:
                                end_ms = start_ms + 1000
                            cues.append(SubtitleCue(len(cues) + 1, start_ms, end_ms, clean_txt))
                        continue
            except Exception:
                pass
            i += 1

    print(f"✅ [SRT] Successfully loaded {len(cues)} valid dialogue cues.")

    # Calculate safe adjusted canvas duration
    last_cue_end_ms = cues[-1].end_ms if cues else 0
    if total_video_ms <= 0:
        adjusted_video_ms = max(5000, last_cue_end_ms + 3000)
        print(f"🎬 [CANVAS] Duration was 00:00:00. Automatically set canvas to {adjusted_video_ms/1000:.1f}s.")
    elif last_cue_end_ms > total_video_ms:
        overhang_sec = (last_cue_end_ms - total_video_ms) / 1000.0
        print(f"⚠️ [CANVAS] Expanding canvas by {overhang_sec:.1f}s to guarantee dialogue is never cut off.")
        adjusted_video_ms = last_cue_end_ms + 3000
    else:
        adjusted_video_ms = total_video_ms

    return cues, adjusted_video_ms


# ==================================================================================================
# 5. KOKORO-82M LOCAL TTS ENGINE (MULTI-LANGUAGE, ZERO EDGE-TTS)
# ==================================================================================================
def get_kokoro_pipeline(lang_code: str = "h") -> Any:
    """Returns or loads a cached KPipeline instance with robust multi-platform auto-install."""
    global _KOKORO_PIPELINES

    lang_code = lang_code.lower()
    if lang_code in _KOKORO_PIPELINES:
        return _KOKORO_PIPELINES[lang_code]

    try:
        from kokoro import KPipeline  # type: ignore
    except ImportError:
        print("⚡ [KOKORO SETUP] 'kokoro' library not found. Installing locally...")
        installed = False
        for pkg in ["kokoro", "git+https://github.com/hexgrad/kokoro.git"]:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg, "soundfile"], check=True)
                from kokoro import KPipeline  # type: ignore
                installed = True
                print(f"✅ [KOKORO SETUP] Installed successfully from {pkg}!")
                break
            except Exception:
                continue

        if not installed:
            raise RuntimeError(
                "Kokoro-82M TTS is not installed in the environment.\n"
                "Please run: pip install kokoro soundfile"
            )

    print(f"📦 [KOKORO TTS] Initializing Kokoro-82M pipeline for language '{lang_code}'...")
    pipeline = KPipeline(lang_code=lang_code)
    _KOKORO_PIPELINES[lang_code] = pipeline
    print(f"✅ [KOKORO TTS] Pipeline ready for language '{lang_code}'.")
    return pipeline


def synthesize_single_cue_kokoro(
    cue: SubtitleCue,
    pipeline: Any,
    voice: str,
    output_wav_path: Path,
    speed: float = 1.0,
) -> Path:
    """Synthesizes dialogue for a single cue using Kokoro-82M and writes a 24kHz WAV file."""
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


# ==================================================================================================
# 6. RVC VOICE CONVERSION ENGINE (RMVPE PITCH EXTRACTION)
# ==================================================================================================
def ensure_rvc_dependencies() -> bool:
    """
    Ensures rvc-python is installed and importable.
    Uses --no-deps to bypass omegaconf/hydra-core dependency resolver conflicts on Hugging Face Spaces.
    """
    global RVCInference, RVC_AVAILABLE
    if RVC_AVAILABLE and RVCInference is not None:
        return True

    try:
        _patch_tensorboard_for_fairseq()
        from rvc_python.infer import RVCInference as _RVCClass  # type: ignore
        RVCInference = _RVCClass
        RVC_AVAILABLE = True
        return True
    except (ImportError, ModuleNotFoundError):
        print("⚡ [RVC SETUP] 'rvc_python' not found in environment. Installing with --no-deps (bypassing omegaconf conflict)...")
        install_commands = [
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "fairseq-fixed", "pyworld-fixed", "rvc-python"],
            [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "rvc-python"],
        ]
        for cmd in install_commands:
            try:
                subprocess.run(cmd, check=True)
                _patch_tensorboard_for_fairseq()
                from rvc_python.infer import RVCInference as _RVCClass  # type: ignore
                RVCInference = _RVCClass
                RVC_AVAILABLE = True
                print("✅ [RVC SETUP] 'rvc-python' installed with --no-deps and verified successfully!")
                return True
            except Exception:
                continue

    except Exception as e:
        print(f"💡 [RVC NOTICE] Could not load RVC ({e}). Falling back to base Kokoro TTS.")

    return False


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
            ]
            if Path("weights").exists():
                candidates.extend(list(Path("weights").glob("*.pth")))
            if Path("models").exists():
                candidates.extend(list(Path("models").rglob("*.pth")))

            for cand in candidates:
                if cand.exists() and cand.stat().st_size > 10_000_000:
                    self.model_path = str(cand.resolve())
                    if not self.index_path:
                        idx_cand = cand.with_suffix(".index")
                        if idx_cand.exists():
                            self.index_path = str(idx_cand.resolve())
                    break

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
        """Initializes the RVC engine with GPU acceleration."""
        global RVCInference, RVC_AVAILABLE
        print(f"\n🎙️ [STEP 2: RVC] Initializing RVC Engine with model: {Path(self.model_path).name}...")
        if not ensure_rvc_dependencies():
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


# ==================================================================================================
# 7. TIME-SYNC & TIMELINE CANVAS CHUNK ASSEMBLY
# ==================================================================================================
def time_sync_audio_file(audio_path: Path, target_dur_ms: int) -> AudioSegment:
    """
    Fits audio chunk to exact subtitle duration:
    - Pitch-preserving time-stretch via phase vocoder (librosa) if audio is too long.
    - Clean silence padding if audio is too short.
    """
    try:
        y, sr = sf.read(str(audio_path), dtype="float32")
        if len(y.shape) > 1:
            y = y.mean(axis=1)

        actual_dur_ms = int(round((len(y) / sr) * 1000.0))

        if actual_dur_ms > target_dur_ms and LIBROSA_AVAILABLE and librosa:
            rate = min(float(actual_dur_ms) / float(target_dur_ms), 2.5)  # Cap compression at 2.5x
            y_stretched = librosa.effects.time_stretch(y, rate=rate)
            stretched_int16 = (np.clip(y_stretched, -1.0, 1.0) * 32767).astype(np.int16)

            seg = AudioSegment(
                data=stretched_int16.tobytes(),
                sample_width=2,
                frame_rate=sr,
                channels=1,
            )
            return seg[:target_dur_ms] if len(seg) > target_dur_ms else seg + AudioSegment.silent(duration=target_dur_ms - len(seg), frame_rate=sr)

        elif actual_dur_ms < target_dur_ms:
            audio_int16 = (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16)
            seg = AudioSegment(
                data=audio_int16.tobytes(),
                sample_width=2,
                frame_rate=sr,
                channels=1,
            )
            return seg + AudioSegment.silent(duration=target_dur_ms - actual_dur_ms, frame_rate=sr)

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
        return seg[:target_dur_ms] if len(seg) > target_dur_ms else seg + AudioSegment.silent(duration=target_dur_ms - len(seg), frame_rate=SAMPLE_RATE)


def concat_audio_chunks_ffmpeg(chunk_paths: List[Path], output_path: Path) -> Path:
    """Concatenates audio chunk files using FFmpeg stream copy (zero-RAM, lossless, instant)."""
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not chunk_paths:
        # Generate safe silence file if no chunk files exist
        silent = AudioSegment.silent(duration=1000, frame_rate=SAMPLE_RATE)
        silent.export(str(output_path), format="wav")
        return output_path

    if len(chunk_paths) == 1:
        shutil.copy2(chunk_paths[0], output_path)
        return output_path

    concat_txt = output_path.parent / f"concat_list_{int(time.time()*1000)}.txt"
    try:
        with open(concat_txt, "w", encoding="utf-8") as f:
            for p in chunk_paths:
                clean_p = str(p.resolve()).replace("\\", "/")
                f.write(f"file '{clean_p}'\n")

        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(concat_txt), "-c", "copy", str(output_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg concat exited with code {res.returncode}")

    except Exception as e:
        print(f"⚠️ [FFMPEG STREAM CONCAT FALLBACK] Soundfile streaming concatenation: {e}")
        # Low-memory streaming concatenation block-by-block (zero OOM risk on 2-hour files)
        with sf.SoundFile(str(output_path), mode='w', samplerate=SAMPLE_RATE, channels=1, subtype='PCM_16') as out_f:
            for p in chunk_paths:
                with sf.SoundFile(str(p), mode='r') as in_f:
                    while True:
                        block = in_f.read(65536, dtype='float32')
                        if len(block) == 0:
                            break
                        out_f.write(block)
    finally:
        if concat_txt.exists():
            try:
                concat_txt.unlink()
            except Exception:
                pass

    return output_path


# ==================================================================================================
# 8. CHUNK-BASED DUBBING PIPELINE (OOM-PROOF FOR 2-HOUR HF SPACES)
# ==================================================================================================
def process_single_language_chunked(
    cues: List[SubtitleCue],
    total_video_ms: int,
    lang_name: str,
    output_wav_path: Path,
    converter: CharacterVoiceConverter,
    work_dir: Path,
    progress_callback=None,
    overall_lang_offset: float = 0.0,
    overall_lang_weight: float = 1.0,
) -> Path:
    """
    Processes a single language in 5-minute chunks:
    1. Splits timeline into 5-minute chunks.
    2. Sequentially executes: Kokoro TTS -> RVC Conversion -> Chunk Audio Assembly.
    3. Flushes each chunk to disk and cleans RAM/VRAM completely before moving to the next.
    4. Merges all chunk files into the final master WAV via FFmpeg.
    """
    lang_meta = SUPPORTED_LANGUAGES.get(lang_name, SUPPORTED_LANGUAGES["Hindi"])
    lang_code = lang_meta["lang_code"]
    voice = lang_meta["voice"]

    pipeline = get_kokoro_pipeline(lang_code)

    num_chunks = max(1, math.ceil(total_video_ms / CHUNK_DURATION_MS))
    print(f"\n📦 [CHUNKING ENGINE] Processing '{lang_name}' across {num_chunks} chunks ({CHUNK_DURATION_MINUTES} mins/chunk)...")

    # Group cues by chunk window
    cues_by_chunk: Dict[int, List[SubtitleCue]] = {idx: [] for idx in range(num_chunks)}
    for cue in cues:
        c_idx = min(max(0, cue.start_ms // CHUNK_DURATION_MS), num_chunks - 1)
        if c_idx in cues_by_chunk:
            cues_by_chunk[c_idx].append(cue)

    chunk_output_files: List[Path] = []
    chunk_temp_dir = work_dir / f"chunks_{lang_code}"
    chunk_temp_dir.mkdir(parents=True, exist_ok=True)

    for c_idx in range(num_chunks):
        c_start_ms = c_idx * CHUNK_DURATION_MS
        c_end_ms = min((c_idx + 1) * CHUNK_DURATION_MS, total_video_ms)
        c_dur_ms = max(1000, c_end_ms - c_start_ms)
        chunk_cues = cues_by_chunk.get(c_idx, [])

        if not chunk_cues:
            # If this 5-minute chunk has no dialogue lines, generate silent canvas to maintain exact timeline sync
            chunk_canvas = AudioSegment.silent(duration=c_dur_ms, frame_rate=SAMPLE_RATE)
            chunk_slice_wav = chunk_temp_dir / f"slice_{c_idx:04d}.wav"
            chunk_canvas.export(str(chunk_slice_wav), format="wav")
            chunk_output_files.append(chunk_slice_wav)
            continue

        print(f"\n🧩 [{lang_name} | Chunk {c_idx + 1}/{num_chunks}] [{c_start_ms//60000}m -> {c_end_ms//60000}m] with {len(chunk_cues)} cues...")

        # Sub-directory for this chunk's temporary audio files
        sub_dir = chunk_temp_dir / f"chunk_{c_idx:04d}"
        sub_dir.mkdir(parents=True, exist_ok=True)

        # 1. Kokoro TTS for this chunk
        for cue in chunk_cues:
            cue_wav = sub_dir / f"cue_{cue.cue_id:04d}_base.wav"
            synthesize_single_cue_kokoro(cue, pipeline, voice=voice, output_wav_path=cue_wav)
            cue.base_wav_path = str(cue_wav)

        # 2. RVC Voice Conversion for this chunk
        for cue in chunk_cues:
            if converter.engine:
                out_rvc_wav = sub_dir / f"cue_{cue.cue_id:04d}_rvc.wav"
                converter.convert_file(Path(cue.base_wav_path), out_rvc_wav)
                cue.rvc_wav_path = str(out_rvc_wav)
            else:
                cue.rvc_wav_path = cue.base_wav_path

        # 3. Assemble onto a clean 5-minute AudioSegment canvas
        chunk_canvas = AudioSegment.silent(duration=c_dur_ms, frame_rate=SAMPLE_RATE)
        for cue in chunk_cues:
            target_wav = Path(cue.rvc_wav_path)
            aligned_seg = time_sync_audio_file(target_wav, cue.target_dur_ms)
            rel_offset = max(0, cue.start_ms - c_start_ms)
            chunk_canvas = chunk_canvas.overlay(aligned_seg, position=rel_offset)

        # 4. Save chunk slice to disk
        chunk_slice_wav = chunk_temp_dir / f"slice_{c_idx:04d}.wav"
        chunk_canvas.export(str(chunk_slice_wav), format="wav")
        chunk_output_files.append(chunk_slice_wav)

        # 5. MEMORY SAFETY: Zero out chunk buffers & force garbage collection
        del chunk_canvas
        shutil.rmtree(sub_dir, ignore_errors=True)
        gc.collect()
        try:
            import torch  # type: ignore
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        # Progress reporting
        if progress_callback:
            chunk_frac = (c_idx + 1) / num_chunks
            current_overall = overall_lang_offset + (chunk_frac * overall_lang_weight)
            progress_callback(
                current_overall,
                desc=f"⚡ [{lang_name}] Chunk {c_idx + 1}/{num_chunks} complete ({get_memory_stats()})",
            )

    # 6. Concatenate all chunk files into final master WAV for this language
    print(f"\n💾 [MERGE] Losslessly concatenating {len(chunk_output_files)} chunk slices into: {output_wav_path.name}...")
    concat_audio_chunks_ffmpeg(chunk_output_files, output_wav_path)

    # Clean chunk temp files to preserve disk space
    shutil.rmtree(chunk_temp_dir, ignore_errors=True)

    print(f"✅ [{lang_name}] Master Dubbed Audio generated successfully: {output_wav_path.resolve()}!")
    return output_wav_path


# ==================================================================================================
# 9. SEQUENTIAL MULTI-LANGUAGE PIPELINE ORCHESTRATOR
# ==================================================================================================
def run_sequential_multi_language_pipeline(
    srt_file: str,
    output_dir: str = "./outputs/dubbed_masters",
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 48,
    selected_languages: Optional[List[str]] = None,
    pitch_shift: int = 0,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Sequentially processes each selected language one-by-one:
    - Language 1 is 100% completed and merged before Language 2 begins.
    - Guarantees zero memory leaks on 2-hour videos.
    """
    pipeline_start = time.time()
    total_video_ms = ((hours * 3600) + (minutes * 60) + seconds) * 1000

    if not selected_languages:
        selected_languages = ["Hindi"]

    out_base = Path(output_dir).resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.gettempdir()) / f"dub_run_{int(time.time())}"
    work_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print("🍥 SEQUENTIAL MULTI-LANGUAGE DUBBING PIPELINE (HUGGING FACE SPACES)")
    print("=" * 75)
    print(f"Subtitles:        {srt_file}")
    print(f"Selected Langs:   {selected_languages}")
    print(f"Target Duration:  {hours:02d}:{minutes:02d}:{seconds:02d} ({total_video_ms/1000:.1f}s)")
    print(f"Memory Status:    {get_memory_stats()}")
    print("=" * 75 + "\n")

    # Step 1: Parse SRT subtitles once
    cues, adjusted_video_ms = parse_srt_file(srt_file, total_video_ms)
    if not cues:
        raise ValueError(f"No valid dialogue cues found in SRT file: {srt_file}")

    # Step 2: Initialize RVC model converter (auto-downloads if needed)
    converter = CharacterVoiceConverter(
        pitch_shift=pitch_shift,
        model_url=CONFIGURED_RVC_MODEL_URL,
    )

    results_by_language: Dict[str, str] = {}
    num_langs = len(selected_languages)

    # Step 3: Sequential Processing Loop
    for lang_idx, lang_name in enumerate(selected_languages):
        lang_start = time.time()
        print(f"\n🌍 ==================================================================")
        print(f"🌍 [LANGUAGE {lang_idx + 1}/{num_langs}] Starting pipeline for: {lang_name.upper()}")
        print(f"🌍 ==================================================================")

        lang_slug = lang_name.lower().replace(" ", "_")
        lang_out_wav = out_base / f"master_dubbed_{lang_slug}_{int(time.time())}.wav"

        lang_offset = lang_idx / num_langs
        lang_weight = 1.0 / num_langs

        process_single_language_chunked(
            cues=cues,
            total_video_ms=adjusted_video_ms,
            lang_name=lang_name,
            output_wav_path=lang_out_wav,
            converter=converter,
            work_dir=work_dir,
            progress_callback=progress_callback,
            overall_lang_offset=lang_offset,
            overall_lang_weight=lang_weight,
        )

        results_by_language[lang_name] = str(lang_out_wav)
        lang_elapsed = time.time() - lang_start
        print(f"🎉 [LANGUAGE {lang_idx + 1}/{num_langs} COMPLETE] Finished {lang_name} in {lang_elapsed:.1f}s ({lang_elapsed/60:.2f} mins).\n")

    # Clean overall temporary working directory
    shutil.rmtree(work_dir, ignore_errors=True)

    if progress_callback:
        progress_callback(1.0, desc="🏆 All Selected Languages Dubbed & Merged Successfully!")

    total_time = time.time() - pipeline_start
    return {
        "results": results_by_language,
        "elapsed_seconds": total_time,
        "cues_count": len(cues),
        "rvc_active": converter.engine is not None,
        "video_duration_ms": adjusted_video_ms,
    }


# ==================================================================================================
# 10. CLEAN MINIMALIST GRADIO WEB UI (EXACTLY 4 INPUTS)
# ==================================================================================================
CUSTOM_CSS = """
.gradio-container {
    max-width: 840px !important;
    margin: 0 auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.studio-header {
    text-align: center;
    padding: 26px 20px;
    background: linear-gradient(135deg, #181824 0%, #101018 100%);
    border-radius: 16px;
    border: 1px solid rgba(255, 140, 0, 0.25);
    margin-bottom: 22px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
}

.studio-title {
    font-size: 2.1rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff8c00 0%, #ff4500 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}

.studio-subtitle {
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


def gradio_multi_dubbing_handler(
    srt_file_obj,
    hours: float,
    minutes: float,
    seconds: float,
    selected_languages: List[str],
    progress=None,
):
    """Gradio handler connecting UI to sequential multi-language chunking pipeline."""
    if progress is None and GRADIO_AVAILABLE and gr is not None:
        try:
            progress = gr.Progress(track_tqdm=True)
        except Exception:
            progress = None

    if srt_file_obj is None:
        return "### ⚠️ Please upload an SRT subtitle file (*.srt) to begin.", None, None

    if not selected_languages:
        return "### ⚠️ Please select at least one target language from the checkboxes.", None, None

    srt_path = getattr(srt_file_obj, "name", str(srt_file_obj))
    out_dir = Path(tempfile.gettempdir()) / "hf_dubbed_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        pipeline_output = run_sequential_multi_language_pipeline(
            srt_file=srt_path,
            output_dir=str(out_dir),
            hours=int(hours or 0),
            minutes=int(minutes or 0),
            seconds=int(seconds or 0),
            selected_languages=selected_languages,
            progress_callback=progress,
        )

        results_dict = pipeline_output["results"]
        elapsed = pipeline_output["elapsed_seconds"]
        rvc_active = pipeline_output["rvc_active"]

        generated_file_paths = list(results_dict.values())
        first_audio_path = generated_file_paths[0] if generated_file_paths else None

        # Build Markdown completion table
        rvc_status_str = "✅ Active (RMVPE Character Model)" if rvc_active else "⚡ Kokoro Neural TTS Baseline"
        table_rows = ""
        for lang, path in results_dict.items():
            dur_sec = sf.info(path).duration
            dur_str = f"{int(dur_sec//60):02d}:{int(dur_sec%60):02d}"
            table_rows += f"| **{lang}** | `{dur_str}` ({dur_sec:.1f}s) | `{Path(path).name}` |\n"

        status_markdown = f"""
### 🎉 Dubbing Completed Successfully!

| Language | Audio Duration | File Name |
| :--- | :--- | :--- |
{table_rows}

| Pipeline Metric | Result |
| :--- | :--- |
| **Voice Conversion (RVC)** | {rvc_status_str} |
| **Dialogue Lines Processed** | {pipeline_output["cues_count"]} cues |
| **Total Processing Time** | **{elapsed:.1f}s** ({elapsed/60:.2f} mins) |

*You can preview the first language below, or download all dubbed `.wav` files.*
"""
        return status_markdown, first_audio_path, generated_file_paths

    except Exception as e:
        err_msg = f"""
### ❌ Dubbing Failed
An error occurred during execution:
```text
{str(e)}
```
*Tip: Ensure your SRT file contains valid timestamps and dialogue.*
"""
        return err_msg, None, None


def build_ui():
    """Constructs the clean, minimal Gradio interface (strictly 4 requested inputs)."""
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

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="AI Video Dubbing Studio (HF Spaces)") as demo:

        gr.HTML(
            """
            <div class="studio-header">
                <div class="studio-title">🎙️ AI Multi-Language Dubbing Studio</div>
                <div class="studio-subtitle">
                    Kokoro-82M TTS + RVC Voice Conversion • 2-Hour OOM-Safe Chunking for Hugging Face Spaces
                </div>
            </div>
            """
        )

        with gr.Group():
            # 1. Single SRT file upload
            srt_input = gr.File(
                label="1. Translated SRT Subtitles (*.srt)",
                file_types=[".srt"],
                file_count="single",
            )

            # 2. Video Duration Inputs (Timeline Canvas Setup)
            gr.Markdown("#### 2. Video Duration (Timeline Canvas Setup)")
            with gr.Row():
                h_input = gr.Number(label="Hours", value=0, precision=0, minimum=0)
                m_input = gr.Number(label="Minutes", value=0, precision=0, minimum=0)
                s_input = gr.Number(label="Seconds", value=48, precision=0, minimum=0)

            # 3. Language Selection Checkboxes (Multiple Choice)
            gr.Markdown("#### 3. Target Dubbing Languages (Sequential Multi-Language)")
            lang_checkboxes = gr.CheckboxGroup(
                choices=list(SUPPORTED_LANGUAGES.keys()),
                value=["Hindi"],
                label="Select one or more languages to dub sequentially",
                info="The pipeline will automatically process each language one after another without crashing.",
            )

            # 4. Action Button
            start_btn = gr.Button(
                "⚡ Start Dubbing",
                variant="primary",
                size="lg",
                elem_classes="btn-start",
            )

        # Output Section
        gr.Markdown("### 🎧 Dubbed Master Audio Outputs")
        status_box = gr.Markdown(
            """
            <div class="status-card">
                <b>Ready to dub.</b> Upload your <code>.srt</code> file, check your desired languages, and click <b>Start Dubbing</b>.
            </div>
            """
        )
        audio_preview = gr.Audio(
            label="Audio Preview (First Completed Language)",
            type="filepath",
            interactive=False,
        )
        download_files = gr.File(
            label="📥 Download All Dubbed Master Audio Files (.wav)",
            file_count="multiple",
        )

        # Connect button event
        start_btn.click(
            fn=gradio_multi_dubbing_handler,
            inputs=[
                srt_input,
                h_input,
                m_input,
                s_input,
                lang_checkboxes,
            ],
            outputs=[
                status_box,
                audio_preview,
                download_files,
            ],
        )

    return demo


# ==================================================================================================
# 11. MODEL CONFIGURATION & APPLICATION LAUNCH
# ==================================================================================================
# ⚙️ USER MODEL CONFIGURATION:
# [INSERT RVC MODEL DOWNLOAD LINK HERE]
CUSTOM_RVC_MODEL_DOWNLOAD_URL = CONFIGURED_RVC_MODEL_URL

def launch_gradio_app(demo_app):
    """
    Robust Gradio launcher that seamlessly handles:
    1. Hugging Face Spaces (native zero-config launch)
    2. Google Colab (public shareable link https://xxxx.gradio.live)
    3. Localhost / Remote Server with automatic fallback
    """
    is_hf_space = os.getenv("SPACE_ID") is not None or os.getenv("SYSTEM") == "spaces"

    # Strategy 1: Hugging Face Spaces native launch (HF handles port/routing internally)
    if is_hf_space:
        try:
            print("🌐 [LAUNCH] Detected Hugging Face Spaces environment. Launching native interface...")
            demo_app.queue().launch()
            return
        except Exception as e:
            print(f"💡 [LAUNCH] HF Space native launch notice: {e}. Retrying with public link...")

    # Strategy 2: Google Colab & Remote environments (share=True creates public https://xxxx.gradio.live link)
    try:
        print("🌐 [LAUNCH] Launching with share=True for Colab / Remote browser access...")
        demo_app.queue().launch(
            share=True,
            show_error=True,
        )
        return
    except Exception as err:
        print(f"⚠️ [LAUNCH] share=True notice ({err}). Retrying with local server...")

    # Strategy 3: Standard local fallback
    try:
        demo_app.queue().launch(
            share=False,
            show_error=True,
        )
    except Exception as err2:
        print(f"⚠️ [LAUNCH] Local launch notice ({err2}). Attempting forced share=True...")
        demo_app.queue().launch(share=True)


if __name__ == "__main__":
    demo = build_ui()
    print("\n" + "=" * 75)
    print("🚀 [DUBBING STUDIO] Launching Interactive Web Interface...")
    print(f"🎙️ Configured RVC Model URL: {CUSTOM_RVC_MODEL_DOWNLOAD_URL}")
    print("=" * 75 + "\n")
    launch_gradio_app(demo)

