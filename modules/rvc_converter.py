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
import traceback as _traceback
from pathlib import Path

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import soundfile as sf
from .tts_generator import SubtitleCue
from .model_downloader import ensure_naruto_model, PTH_CANONICAL_NAME, INDEX_CANONICAL_NAME

# --------------------------------------------------------------------------------------------------
# Patch TensorBoard / TensorFlow compatibility shim for FairSeq & RVC
# --------------------------------------------------------------------------------------------------
def _patch_tensorboard_for_fairseq():
    """
    Prevents `ImportError: cannot import name 'notf' from 'tensorboard.compat'`
    which occurs in Python 3.13 / Ubuntu 24.04 Colab when fairseq imports torch.utils.tensorboard.
    Neither FairSeq Hubert extraction nor RVC inference uses TensorBoard.
    """
    import sys
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

def _patch_fairseq_registry():
    """Prevents FairSeq setup_registry NoneType unpacking crash."""
    for mod_name in list(sys.modules.keys()):
        if mod_name == "fairseq" or mod_name.startswith("fairseq."):
            sys.modules.pop(mod_name, None)

_patch_fairseq_registry()
_patch_tensorboard_for_fairseq()

try:
    import torch
    TORCH_AVAILABLE = True
    # In PyTorch 2.6+, torch.load defaults to weights_only=True which breaks RVC checkpoints
    _orig_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

# ── Try importing RVC (infer-rvc-python first [fairseq-free], rvc-python as legacy fallback) ──────
RVCInference = None
RVC_AVAILABLE = False
_patch_fairseq_registry()
_patch_tensorboard_for_fairseq()
for _rvc_mod, _rvc_cls in [
    ("infer_rvc_python.infer", "RVCInference"),   # preferred: fairseq-free fork
    ("rvc_python.infer",       "RVCInference"),   # legacy: rvc-python (needs fairseq)
]:
    try:
        import importlib as _il
        _m = _il.import_module(_rvc_mod)
        RVCInference = getattr(_m, _rvc_cls)
        RVC_AVAILABLE = True
        print(f"✅ [RVC Module] Loaded from '{_rvc_mod}'.")
        break
    except Exception:
        continue
if not RVC_AVAILABLE:
    print("⚠️ [RVC Module] No RVC library found — RVCBatchConverter will raise on init.")



def find_default_naruto_model() -> Tuple[Optional[str], Optional[str]]:
    """
    Locates existing Naruto RVC model and index across project and standard RVC paths.
    Returns (model_path, index_path) or (None, None).
    """
    root = Path.cwd().resolve()
    pth_candidates = [
        root / "models" / "naruto" / PTH_CANONICAL_NAME,
        root / "models" / "naruto" / "naruto-uzumaki-by-mboisuper.pth",
        root / "weights" / "naruto-uzumaki-by-mboisuper.pth",
        root / "weights" / PTH_CANONICAL_NAME,
    ]
    index_candidates = [
        root / "models" / "naruto" / INDEX_CANONICAL_NAME,
        root / "models" / "naruto" / "added_IVF102_Flat_nprobe_1_naruto-uzumaki-by-mboisuper_v2.index",
        root / "logs" / "naruto" / "added_IVF102_Flat_nprobe_1_naruto-uzumaki-by-mboisuper_v2.index",
        root / "logs" / "naruto" / INDEX_CANONICAL_NAME,
    ]

    found_pth = next((str(p) for p in pth_candidates if p.exists() and p.stat().st_size > 10_000_000), None)
    found_index = next((str(p) for p in index_candidates if p.exists() and p.stat().st_size > 1_000_000), None)
    return found_pth, found_index


class RVCBatchConverter:
    """
    RVC Batch Voice Conversion Engine.
    Loads Naruto or target character voice model once and processes audio chunks in batches.
    Supports auto-downloading and automatic feature index location.
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
        auto_download: bool = True,
    ):
        """
        Initialize RVC Engine.

        Args:
            model_path: Path to the character .pth file. If None or not found, auto-resolves/downloads.
            index_path: Path to the feature .index file. If None, auto-searches known folders.
            pitch_shift: Semitone pitch shift (0 for normal, +12 for octave up, -12 down).
            f0_method: Pitch extraction algorithm ('rmvpe', 'pm', 'harvest', 'crepe').
            index_rate: Strength of character timbre retrieval (0.0 to 1.0, default 0.75).
            protect: Voiceless consonant and breath protection (0.0 to 0.5, default 0.33).
            filter_radius: Median filter radius for pitch recognition (default 3).
            resample_sr: Target sample rate (default 44100 Hz).
            device: 'cuda:0' or 'cpu'. Automatically chooses cuda if available.
            auto_download: Whether to automatically download Naruto weights if missing.
        """
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

        # Resolve model and index paths
        resolved_pth, resolved_index = self._resolve_model_files(model_path, index_path, auto_download)
        self.model_path = resolved_pth
        self.index_path = resolved_index

        if self.model_path and os.path.exists(self.model_path):
            self.load_model(self.model_path, self.index_path)

    def _resolve_model_files(
        self,
        model_path: Optional[str],
        index_path: Optional[str],
        auto_download: bool,
    ) -> Tuple[Optional[str], Optional[str]]:
        """Resolves existing or downloaded model and index paths."""
        resolved_pth = model_path
        resolved_index = index_path

        # If model_path is not specified or doesn't exist, search local candidates
        if not resolved_pth or not os.path.exists(resolved_pth):
            found_pth, found_idx = find_default_naruto_model()
            if found_pth:
                resolved_pth = found_pth
                if not resolved_index:
                    resolved_index = found_idx
            elif auto_download:
                print("🍥 [RVC ENGINE] Model weights not found locally. Initiating auto-download...")
                download_res = ensure_naruto_model()
                resolved_pth = download_res.get("model_path")
                if not resolved_index:
                    resolved_index = download_res.get("index_path")

        # If index_path was still not found, search adjacent or standard directories
        if resolved_pth and (not resolved_index or not os.path.exists(resolved_index)):
            p = Path(resolved_pth)
            # Check adjacent .index files
            adjacent_indices = list(p.parent.glob("*.index"))
            if adjacent_indices:
                resolved_index = str(adjacent_indices[0])
            else:
                # Check standard logs/naruto/
                logs_dir = Path.cwd() / "logs" / "naruto"
                if logs_dir.exists():
                    log_indices = list(logs_dir.glob("*.index"))
                    if log_indices:
                        resolved_index = str(log_indices[0])

        return resolved_pth, resolved_index

    def load_model(self, model_path: str, index_path: Optional[str] = None):
        """Loads RVC weights onto the selected device."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RVC model .pth not found: {model_path}")

        self.model_path = model_path
        self.index_path = index_path

        print("\n" + "=" * 65)
        print("🍥 [RVC ENGINE] Loading Target Character Voice Model...")
        print(f"📦 Model File:  {Path(model_path).name}")
        print(f"📑 Index File:  {Path(index_path).name if index_path and os.path.exists(index_path) else 'None (auto-search/none)'}")
        print(f"⚡ Device:      {self.device} (RMVPE Pitch Extraction)")
        print(f"🎵 Pitch Shift: {self.pitch_shift:+d} semitones | Index Rate: {self.index_rate}")
        print("=" * 65)

        if not RVC_AVAILABLE:
            raise RuntimeError(
                "\n" + "!" * 80 + "\n"
                "🚨 [RVC FATAL] 'rvc-python' is NOT installed or failed to import!\n"
                "   Silent passthrough / fallback is permanently DISABLED.\n"
                "   Run the Colab Step-2 cell to install all dependencies, then restart the runtime.\n"
                + "!" * 80 + "\n"
            )

        try:
            # Instantiate persistent RVC Inference engine
            self.engine = RVCInference(device=self.device)
            self.engine.load_model(self.model_path, index_path=self.index_path or "", version="v2")
            if hasattr(self.engine, "set_params"):
                self.engine.set_params(
                    f0method="rmvpe",
                    f0up_key=self.pitch_shift,
                    index_rate=self.index_rate,
                    protect=self.protect,
                    filter_radius=self.filter_radius,
                    resample_sr=self.resample_sr,
                )
            self._is_ready = True
            print("✅ [RVC ENGINE] Model loaded successfully into memory!\n")

        except Exception as e:
            import traceback
            print("\n" + "!" * 70)
            print(f"🚨 [RVC LOAD ERROR] Failed to load RVC engine: {e}")
            traceback.print_exc()
            print("!" * 70 + "\n")
            self.engine = None
            self._is_ready = False

    def convert_single_file(self, input_wav: str, output_wav: str) -> bool:
        """
        Converts a single audio file using RVC with Mono Audio Handshake and rmvpe.
        Silent passthrough is permanently DISABLED — raises RuntimeError if engine is not ready.
        """
        if not self._is_ready or not self.engine:
            raise RuntimeError(
                f"🚨 [RVC ERROR] Cannot convert '{Path(input_wav).name}': RVC Engine is not initialized!\n"
                "   Silent passthrough / fallback is permanently DISABLED."
            )

        clean_temp_wav = Path(input_wav).with_name(f"{Path(input_wav).stem}_mono16k.wav")
        try:
            # Audio Handshake: enforce 1-channel Mono & 16000Hz sampling rate
            y, sr = sf.read(str(input_wav), dtype="float32")
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)

            target_sr = 16000
            try:
                import librosa
                if sr != target_sr:
                    y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr
            except Exception:
                pass

            sf.write(str(clean_temp_wav), y, sr, subtype="PCM_16")

            if hasattr(self.engine, "set_params"):
                self.engine.set_params(
                    f0method="rmvpe",
                    f0up_key=self.pitch_shift,
                    index_rate=self.index_rate,
                    protect=self.protect,
                )
            try:
                self.engine.infer_file(str(clean_temp_wav), str(output_wav))
            except TypeError:
                self.engine.infer_file(
                    input_path=str(clean_temp_wav),
                    output_path=str(output_wav),
                    pitch=self.pitch_shift,
                    f0method="rmvpe",
                )
            return True
        except Exception as e:
            print("\n" + "!" * 70)
            print(f"🚨 [RVC CONVERSION ERROR] Failed for {Path(input_wav).name}: {e}")
            _traceback.print_exc()
            print("!" * 70 + "\n")
            # NO silent copyfile fallback — re-raise so caller sees the real error
            raise RuntimeError(f"RVC audio conversion failed on {Path(input_wav).name}: {e}")
        finally:
            if clean_temp_wav.exists():
                clean_temp_wav.unlink(missing_ok=True)

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


def convert_base_audio_to_naruto(
    audio_inputs: Union[List[str], str, List[Path]],
    output_dir: str = "./outputs/naruto_rvc",
    pitch_shift: int = 0,
    f0_method: str = "rmvpe",
    model_path: Optional[str] = None,
    index_path: Optional[str] = None,
    progress_callback=None,
) -> List[str]:
    """
    Convenience function: Takes base audio files (generated by Edge-TTS for Hindi, Spanish,
    French, Portuguese, etc.) and converts their timbre into Naruto's voice using the loaded model.
    Outputs the final converted .wav files into the designated output folder.

    Args:
        audio_inputs: Path to a single .wav, directory containing .wav files, or a list of .wav paths.
        output_dir: Output folder destination for converted Naruto audio files.
        pitch_shift: Semitones transposition (0=natural, +12=octave up).
        f0_method: Pitch extraction algorithm ('rmvpe' default).
        model_path: Optional custom .pth path. If None, auto-resolves/downloads Naruto V2.
        index_path: Optional custom .index path. If None, auto-locates feature index.
        progress_callback: Optional progress reporter callback.

    Returns:
        List of generated Naruto .wav file paths.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve inputs
    if isinstance(audio_inputs, (str, Path)):
        p = Path(audio_inputs).resolve()
        if p.is_dir():
            file_list = sorted([str(f) for f in p.glob("*.wav")])
        elif p.is_file():
            file_list = [str(p)]
        else:
            raise FileNotFoundError(f"Input audio path not found: {audio_inputs}")
    else:
        file_list = [str(Path(f).resolve()) for f in audio_inputs]

    if not file_list:
        print(f"⚠️ [NARUTO RVC] No WAV audio files found to convert in: {audio_inputs}")
        return []

    print("\n" + "=" * 65)
    print(f"🍥 [NARUTO RVC BATCH] Converting {len(file_list)} base audio files to Naruto voice...")
    print(f"📁 Output Folder: {out_dir}")
    print("=" * 65)

    converter = RVCBatchConverter(
        model_path=model_path,
        index_path=index_path,
        pitch_shift=pitch_shift,
        f0_method=f0_method,
        auto_download=True,
    )

    converted_paths = converter.convert_batch(
        cues_or_files=file_list,
        output_dir=str(out_dir),
        progress_callback=progress_callback,
    )

    return converted_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Naruto RVC V2 Batch Voice Converter (Hindi, Spanish, French, Portuguese, etc.)"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input WAV file or directory containing base audio WAVs",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./outputs/naruto_rvc",
        help="Designated folder for final converted .wav files",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=None,
        help="Path to Naruto .pth file (auto-downloads if omitted/not found)",
    )
    parser.add_argument(
        "--index",
        type=str,
        default=None,
        help="Path to Naruto .index file (auto-discovered if omitted)",
    )
    parser.add_argument(
        "--pitch", "-p",
        type=int,
        default=0,
        help="Pitch transpose in semitones (0 for natural, +12 for octave up)",
    )
    parser.add_argument(
        "--f0-method",
        type=str,
        default="rmvpe",
        choices=["rmvpe", "pm", "harvest", "crepe"],
        help="Pitch extraction algorithm (RMVPE recommended)",
    )

    args = parser.parse_args()

    results = convert_base_audio_to_naruto(
        audio_inputs=args.input,
        output_dir=args.output_dir,
        pitch_shift=args.pitch,
        f0_method=args.f0_method,
        model_path=args.model,
        index_path=args.index,
    )
    print(f"\n🎉 Successfully processed {len(results)} files into '{args.output_dir}'!")
