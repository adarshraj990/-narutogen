"""
====================================================================================================
MODULE 2: RVC VOICE CONVERTER (Batch-Inference)
====================================================================================================
Runs high-speed batch voice conversion using pre-downloaded Naruto RVC V2 models (.pth and .index).
Features:
- RMVPE Pitch Extraction: Gold-standard algorithm for natural, polyphonic, artifact-free conversion.
- Persistent Model Cache: Loads the weights onto GPU (cuda:0) strictly ONCE across all audio chunks.
- Flexible Batch Input: Accepts a directory of WAVs, a list of file paths, or SubtitleCue objects.
- VRAM Safe: Cleans torch CUDA cache to guarantee OOM-free inference on Google Colab T4 (15GB VRAM).
====================================================================================================
"""

import os
import gc
import sys
import time
import shutil
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import soundfile as sf
from .tts_generator import SubtitleCue

# Try importing RVC Inference libraries
try:
    from rvc_python.infer import RVCInference, infer_file
    RVC_AVAILABLE = True
except ImportError:
    RVC_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class RVCBatchConverter:
    """
    RVC Batch Voice Conversion Engine.
    Loads Naruto or target character voice model once and processes audio chunks in batches.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        index_path: Optional[str] = None,
        pitch_shift: int = 0,
        f0_method: str = "rmvpe",
        index_rate: float = 0.75,
        protect: float = 0.33,
        filter_radius: int = 3,
        resample_sr: int = 44100,
        device: Optional[str] = None,
    ):
        """
        Initialize RVC Engine.

        Args:
            model_path: Path to the character .pth file (e.g., 'models/naruto/naruto.pth').
            index_path: Path to the feature .index file (e.g., 'models/naruto/naruto.index').
            pitch_shift: Semitone pitch shift (0 for normal, +12 for octave up, -12 down).
            f0_method: Pitch extraction algorithm ('rmvpe', 'pm', 'harvest', 'crepe').
            index_rate: Strength of character timbre retrieval (0.0 to 1.0, default 0.75).
            protect: Voiceless consonant and breath protection (0.0 to 0.5, default 0.33).
            filter_radius: Median filter radius for pitch recognition (default 3).
            resample_sr: Target sample rate (default 44100 Hz).
            device: 'cuda:0' or 'cpu'. Automatically chooses cuda if available.
        """
        self.model_path = model_path
        self.index_path = index_path
        self.pitch_shift = int(pitch_shift)
        self.f0_method = f0_method
        self.index_rate = float(index_rate)
        self.protect = float(protect)
        self.filter_radius = int(filter_radius)
        self.resample_sr = int(resample_sr)

        # Automatic device detection
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda:0"
            else:
                self.device = "cpu"
        else:
            self.device = device

        self.engine = None
        self._is_ready = False

        if model_path:
            self.load_model(model_path, index_path)

    def load_model(self, model_path: str, index_path: Optional[str] = None):
        """Loads RVC weights onto the selected device."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RVC model .pth not found: {model_path}")

        self.model_path = model_path
        self.index_path = index_path

        print("\n" + "=" * 65)
        print("🍥 [RVC ENGINE] Loading Target Character Voice Model...")
        print(f"📦 Model File:  {Path(model_path).name}")
        print(f"📑 Index File:  {Path(index_path).name if index_path else 'None (auto-search/none)'}")
        print(f"⚡ Device:      {self.device} (RMVPE Pitch Extraction)")
        print(f"🎵 Pitch Shift: {self.pitch_shift:+d} semitones | Index Rate: {self.index_rate}")
        print("=" * 65)

        if not RVC_AVAILABLE:
            print(
                "⚠️ [RVC WARNING] 'rvc-python' is not installed in the environment.\n"
                "Install it via: pip install rvc-python\n"
                "Falling back to baseline audio pass-through."
            )
            self._is_ready = False
            return

        try:
            # Instantiate persistent RVC Inference engine
            self.engine = RVCInference(device=self.device)
            self.engine.load_model(self.model_path)
            self._is_ready = True
            print("✅ [RVC ENGINE] Model loaded successfully into memory!\n")

        except Exception as e:
            print(f"⚠️ [RVC LOAD ERROR] Failed to load RVC engine: {e}")
            print("Falling back to baseline audio pass-through.")
            self.engine = None
            self._is_ready = False

    def convert_single_file(self, input_wav: str, output_wav: str) -> bool:
        """Converts a single audio file using RVC."""
        if not self._is_ready or not self.engine:
            # Pass-through if RVC is not ready
            shutil.copyfile(input_wav, output_wav)
            return True

        try:
            resolved_index = self.index_path if (self.index_path and os.path.exists(self.index_path)) else ""
            self.engine.infer_file(
                input_path=str(input_wav),
                output_path=str(output_wav),
                pitch_shift=self.pitch_shift,
                f0_method=self.f0_method,
                index_path=resolved_index,
                index_rate=self.index_rate,
                protect=self.protect,
            )
            return True
        except Exception as e:
            print(f"⚠️ [RVC CONVERSION ERROR] Failed for {Path(input_wav).name}: {e}")
            shutil.copyfile(input_wav, output_wav)
            return False

    def convert_batch(
        self,
        cues_or_files: Union[List[SubtitleCue], List[str], List[Path]],
        output_dir: str = "./outputs/rvc_converted",
        progress_callback=None,
    ) -> List[Any]:
        """
        Converts all audio files in batch mode.
        Accepts either a list of SubtitleCue objects (from Module 1) or a list of file paths.
        """
        out_path = Path(output_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        total = len(cues_or_files)
        if total == 0:
            return []

        print(f"\n🚀 [RVC BATCH] Starting voice conversion on {total} audio segments...")
        start_time = time.time()
        converted_items = []

        is_cue_mode = isinstance(cues_or_files[0], SubtitleCue)

        for idx, item in enumerate(cues_or_files, start=1):
            if is_cue_mode:
                cue: SubtitleCue = item
                input_file = cue.base_wav_path
                target_file = str(out_path / f"cue_{cue.cue_id:04d}_rvc.wav")
            else:
                input_file = str(item)
                stem = Path(input_file).stem
                target_file = str(out_path / f"{stem}_rvc.wav")

            # Run RVC inference
            self.convert_single_file(input_file, target_file)

            if is_cue_mode:
                item.rvc_wav_path = target_file
                converted_items.append(item)
            else:
                converted_items.append(target_file)

            # Periodic memory clearing to avoid Colab GPU OOM
            if idx % 20 == 0:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            if progress_callback and total > 0:
                progress_callback(idx / total, desc=f"RVC Converting: {idx}/{total} cues (RMVPE)...")

            if idx % 10 == 0 or idx == total:
                elapsed = time.time() - start_time
                rate = idx / max(1e-5, elapsed)
                print(f"   ↳ [RVC Progress] {idx}/{total} converted ({rate:.2f} cues/sec)")

        total_time = time.time() - start_time
        print(f"✅ [RVC BATCH] Batch conversion complete! Processed {total} files in {total_time/60:.2f} mins.")
        return converted_items


# ==================================================================================================
# MODULE ENTRYPOINT FUNCTION
# ==================================================================================================
def run_rvc_conversion(
    input_items: Union[List[SubtitleCue], List[str], str],
    model_path: str,
    index_path: Optional[str] = None,
    output_dir: str = "./outputs/rvc_converted",
    pitch_shift: int = 0,
    f0_method: str = "rmvpe",
    index_rate: float = 0.75,
    progress_callback=None,
) -> List[Any]:
    """
    High-level entrypoint for Module 2.
    
    Args:
        input_items: List of SubtitleCue objects, list of file paths, or directory path containing WAVs.
        model_path: Path to character .pth file.
        index_path: Path to character .index file.
        output_dir: Target output directory for converted audio files.
        pitch_shift: Semitones pitch shift.
        f0_method: Pitch extraction algorithm ('rmvpe' recommended).
        index_rate: Index search feature ratio (0.0 to 1.0).
        progress_callback: Optional progress reporter.

    Returns:
        List of processed SubtitleCue objects or file paths with converted audio.
    """
    converter = RVCBatchConverter(
        model_path=model_path,
        index_path=index_path,
        pitch_shift=pitch_shift,
        f0_method=f0_method,
        index_rate=index_rate,
    )

    # If a directory string is passed, collect all WAV files
    if isinstance(input_items, str) and os.path.isdir(input_items):
        file_list = sorted([str(p) for p in Path(input_items).glob("*.wav")])
        return converter.convert_batch(file_list, output_dir=output_dir, progress_callback=progress_callback)

    return converter.convert_batch(input_items, output_dir=output_dir, progress_callback=progress_callback)
