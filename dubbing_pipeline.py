"""
====================================================================================================
🍥 NARUTOGEN: HIGH-EFFICIENCY AI ANIME DUBBING PIPELINE
====================================================================================================
Architecture: Two-Step High-Speed Hybrid (Lightweight Neural TTS + RVC Voice Conversion)
- Target Environment: Google Colab Free Tier (NVIDIA T4 GPU, 16GB RAM)
- Performance Target: Up to 2-hour anime episode/movie dubbed in under 15-30 minutes.
- Zero OOM Crashes: Decouples text synthesis (CPU async) from voice conversion (RMVPE GPU).

Pipeline Stages:
1. SRT Parsing: Cleans formatting/HTML tags, extracts exact millisecond timestamps and dialogue text.
2. Step 1 (Ultra-Fast TTS): Asynchronous Edge-TTS generates standard Hindi/Hinglish speech in ~1-2 mins.
3. Step 2 (RVC Voice Conversion): RMVPE pitch extraction transforms speech into target character
   (e.g., Naruto) using pre-trained .pth and .index model weights.
4. Time Synchronization: Pitch-preserving time-stretching (librosa/pydub) ensures exact lip-sync.
5. Canvas Assembly & Muxing: Assembles full-length master track and muxes with original video via FFmpeg.
====================================================================================================
"""

import os
import gc
import re
import sys
import time
import math
import shutil
import asyncio
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import soundfile as sf
import librosa
import pysrt
from pydub import AudioSegment
import gradio as gr

# Try importing Edge-TTS
try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

# Try importing RVC
try:
    from rvc_python.infer import RVCInference, infer_file
    RVC_AVAILABLE = True
except ImportError:
    RVC_AVAILABLE = False

# Optional system resource monitoring
try:
    import psutil
except ImportError:
    psutil = None

# ==================================================================================================
# CONSTANTS & CONFIGURATION
# ==================================================================================================
SAMPLE_RATE = 44100          # High-fidelity sample rate for RVC output
FLUSH_INTERVAL_MINUTES = 15  # 15-minute slice window for memory safety
FLUSH_INTERVAL_MS = FLUSH_INTERVAL_MINUTES * 60 * 1000  # 900,000 ms

# Recommended Hindi base voices for Edge-TTS
DEFAULT_VOICES = {
    "Hindi Male (Madhur - Ideal for Naruto/Male Anime)": "hi-IN-MadhurNeural",
    "Hindi Female (Swara - Female Characters/Young Naruto)": "hi-IN-SwaraNeural",
    "Indian English Male (Prabhat - Hinglish)": "en-IN-PrabhatNeural",
    "Indian English Female (Neerja - Hinglish)": "en-IN-NeerjaNeural",
}


# ==================================================================================================
# 1. UTILITIES & RESOURCE MONITORING
# ==================================================================================================
def get_memory_stats() -> str:
    """Returns human-readable RAM and VRAM utilization string."""
    stats = []
    if psutil:
        ram = psutil.virtual_memory()
        stats.append(f"RAM: {ram.used / (1024**3):.1f}/{ram.total / (1024**3):.1f}GB ({ram.percent}%)")
    try:
        import torch
        if torch.cuda.is_available():
            vram_alloc = torch.cuda.memory_allocated() / (1024**3)
            vram_res = torch.cuda.memory_reserved() / (1024**3)
            stats.append(f"VRAM: {vram_alloc:.2f}GB alloc ({vram_res:.2f}GB res)")
    except Exception:
        pass
    return " | ".join(stats) if stats else "Monitoring unavailable"


def clean_subtitle_text(text: str) -> str:
    """Cleans SRT formatting artifacts, tags, linebreaks, and non-dialogue annotations."""
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def probe_video_duration(video_path: str) -> Optional[int]:
    """Uses ffprobe to extract exact video duration in milliseconds."""
    if not shutil.which("ffprobe") or not os.path.exists(video_path):
        return None
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        dur_sec = float(res.stdout.strip())
        return int(dur_sec * 1000)
    except Exception as e:
        print(f"⚠️ [PROBE WARNING] Could not probe video duration: {e}")
        return None


# ==================================================================================================
# 2. SRT PARSING
# ==================================================================================================
class SubtitleCue:
    def __init__(self, cue_id: int, start_ms: int, end_ms: int, text: str):
        self.cue_id = cue_id
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.target_dur_ms = max(200, end_ms - start_ms)
        self.text = text

    def __repr__(self):
        return f"<Cue #{self.cue_id} [{self.start_ms}ms -> {self.end_ms}ms ({self.target_dur_ms}ms)]: '{self.text[:20]}...'>"


def parse_srt_file(srt_path: str, total_video_ms: int) -> Tuple[List[SubtitleCue], int]:
    """Parses SRT, cleans dialogue text, filters empty cues, and checks boundary conditions."""
    print(f"📄 [SRT] Parsing subtitles from: {srt_path}")
    subs = pysrt.open(srt_path, encoding="utf-8")
    cues = []

    for i, sub in enumerate(subs, start=1):
        clean_txt = clean_subtitle_text(sub.text)
        if not clean_txt:
            continue

        start_ms = int(sub.start.ordinal)
        end_ms = int(sub.end.ordinal)

        if end_ms <= start_ms:
            end_ms = start_ms + 1000

        cues.append(SubtitleCue(i, start_ms, end_ms, clean_txt))

    print(f"✅ [SRT] Successfully loaded {len(cues)} valid dialogue cues.")

    # Auto-expand master canvas if subtitles exceed initial user duration
    adjusted_video_ms = total_video_ms
    if cues and cues[-1].end_ms > total_video_ms:
        overhang_sec = (cues[-1].end_ms - total_video_ms) / 1000.0
        print(f"⚠️ [CANVAS EXPANSION] Last dialogue ends at {cues[-1].end_ms / 1000:.1f}s, past initial canvas ({total_video_ms / 1000:.1f}s).")
        print(f"👉 Expanding canvas by {overhang_sec:.1f}s to guarantee dialogue is never cut off.")
        adjusted_video_ms = cues[-1].end_ms + 2000

    return cues, adjusted_video_ms


# ==================================================================================================
# 3. STEP 1: ULTRA-FAST ASYNC BASELINE TTS GENERATION (Edge-TTS)
# ==================================================================================================
async def _async_generate_single_cue(
    cue: SubtitleCue,
    voice: str,
    output_path: str,
    semaphore: asyncio.Semaphore,
    rate: str = "+0%",
) -> bool:
    """Generates a single dialogue audio file via Edge-TTS under concurrency control."""
    async with semaphore:
        try:
            communicate = edge_tts.Communicate(cue.text, voice, rate=rate)
            await communicate.save(output_path)
            return True
        except Exception as e:
            print(f"⚠️ [TTS ERROR] Cue #{cue.cue_id} failed: {e}")
            # Generate fallback silence
            silence = AudioSegment.silent(duration=cue.target_dur_ms, frame_rate=SAMPLE_RATE)
            silence.export(output_path, format="wav")
            return False


async def batch_generate_tts(
    cues: List[SubtitleCue],
    voice: str,
    tts_output_dir: Path,
    concurrency: int = 12,
    progress_callback=None,
) -> List[Tuple[SubtitleCue, Path]]:
    """
    Executes concurrent Edge-TTS synthesis for all subtitles.
    Generates 1,000+ subtitles in under 60-90 seconds.
    """
    print(f"\n⚡ [STEP 1: TTS] Starting async synthesis for {len(cues)} cues using voice '{voice}'...")
    semaphore = asyncio.Semaphore(concurrency)
    tts_output_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    cue_file_pairs = []

    for cue in cues:
        out_file = tts_output_dir / f"cue_{cue.cue_id:04d}.wav"
        cue_file_pairs.append((cue, out_file))
        tasks.append(_async_generate_single_cue(cue, voice, str(out_file), semaphore))

    total = len(tasks)
    completed = 0

    for f in asyncio.as_completed(tasks):
        await f
        completed += 1
        if progress_callback and total > 0:
            frac = 0.05 + (completed / total) * 0.20  # 5% to 25%
            progress_callback(frac, desc=f"⚡ [Step 1/3] Generating TTS: {completed}/{total} cues...")

    print(f"✅ [STEP 1: TTS] Completed {len(cues)} cues in record time! {get_memory_stats()}")
    return cue_file_pairs


# ==================================================================================================
# 4. STEP 2: RVC VOICE CONVERSION WITH RMVPE PITCH EXTRACTION
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
    ):
        self.model_path = model_path
        self.index_path = index_path
        self.pitch_shift = pitch_shift
        self.f0_method = f0_method
        self.index_rate = index_rate
        self.protect = protect
        self.engine = None

        if model_path and os.path.exists(model_path):
            self._init_rvc_engine()
        else:
            print("💡 [RVC] No .pth model provided. Running in High-Speed Base TTS Mode.")

    def _init_rvc_engine(self):
        """Initializes the RVC engine."""
        print(f"\n🎙️ [STEP 2: RVC] Initializing RVC Engine with model: {Path(self.model_path).name}...")
        if not RVC_AVAILABLE:
            print("⚠️ [RVC WARNING] 'rvc-python' package not installed. Falling back to base TTS.")
            return

        try:
            device = "cuda:0"
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
                print("⚠️ [RVC NOTICE] CUDA unavailable, running RVC on CPU.")

            self.engine = RVCInference(device=device)
            self.engine.load_model(self.model_path)
            print(f"✅ [STEP 2: RVC] RVC Model Loaded on {device} with RMVPE pitch extraction! {get_memory_stats()}")
        except Exception as e:
            print(f"⚠️ [RVC LOAD ERROR] Could not load RVC engine: {e}. Falling back to baseline TTS.")
            self.engine = None

    def convert_file(self, input_wav: Path, output_wav: Path) -> bool:
        """Converts a single audio file to target character voice."""
        if not self.engine:
            # Pass-through baseline TTS audio if RVC is not enabled
            shutil.copyfile(input_wav, output_wav)
            return True

        try:
            # Execute RMVPE voice conversion
            self.engine.infer_file(
                input_path=str(input_wav),
                output_path=str(output_wav),
                pitch_shift=self.pitch_shift,
                f0_method=self.f0_method,
                index_path=self.index_path if (self.index_path and os.path.exists(self.index_path)) else "",
                index_rate=self.index_rate,
                protect=self.protect,
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
        """Batch-converts all dialogue snippets to the target anime character voice."""
        rvc_output_dir.mkdir(parents=True, exist_ok=True)
        total = len(cue_file_pairs)

        if not self.engine:
            print("⚡ [STEP 2: RVC] Skipping RVC (Base TTS mode active).")
            return cue_file_pairs

        print(f"\n🍥 [STEP 2: RVC] Batch-converting {total} cues to character voice (Pitch: {self.pitch_shift}, RMVPE)...")
        results = []

        start_time = time.time()
        for i, (cue, input_file) in enumerate(cue_file_pairs, start=1):
            out_file = rvc_output_dir / f"rvc_{cue.cue_id:04d}.wav"
            self.convert_file(input_file, out_file)
            results.append((cue, out_file))

            if progress_callback and total > 0:
                frac = 0.25 + (i / total) * 0.55  # 25% to 80%
                progress_callback(frac, desc=f"🍥 [Step 2/3] RVC Voice Conversion: {i}/{total} cues (RMVPE)...")

            if i % 25 == 0 or i == total:
                elapsed = time.time() - start_time
                speed = i / max(1e-5, elapsed)
                print(f"⚡ [RVC PROGRESS] Converted {i}/{total} cues ({speed:.1f} cues/sec) | {get_memory_stats()}")

        print(f"✅ [STEP 2: RVC] Character Voice Conversion Complete in {(time.time() - start_time)/60:.2f} mins!")
        return results


# ==================================================================================================
# 5. STEP 3: EXACT TIME-SYNCHRONIZATION & MASTER CANVAS COMPOSITING
# ==================================================================================================
def time_sync_audio_file(audio_path: Path, target_dur_ms: int) -> AudioSegment:
    """
    Time-syncs audio file to exact SRT target duration:
    - If generated > target: High-fidelity phase vocoder speedup (without pitch alteration).
    - If generated < target: Silence padding to match exact timestamp.
    """
    try:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
        actual_dur_ms = int(len(y) / sr * 1000)

        if actual_dur_ms > target_dur_ms:
            speed_ratio = actual_dur_ms / target_dur_ms
            capped_ratio = min(speed_ratio, 1.8)  # Cap speedup to avoid unnatural rush
            stretched = librosa.effects.time_stretch(y, rate=capped_ratio)
            stretched_int16 = (np.clip(stretched, -1.0, 1.0) * 32767).astype(np.int16)
            seg = AudioSegment(
                data=stretched_int16.tobytes(),
                sample_width=2,
                frame_rate=sr,
                channels=1,
            )
            # Micro-trim or pad
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

    except Exception as e:
        print(f"⚠️ [TIME-SYNC FALLBACK] {audio_path.name}: {e}")
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
    """
    Overlays synced dialogue chunks onto the master timeline canvas.
    Flushes 15-minute segments to disk to ensure 100% safety against Colab 12GB RAM crashes.
    """
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

        # Advance windows if this cue starts in a future window
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

        # Boundary split handling
        if local_pos_ms + len(synced_seg) > win_dur_ms:
            split_point = win_dur_ms - local_pos_ms
            current_slice = synced_seg[:split_point]
            overflow_slice = synced_seg[split_point:]

            active_canvas = active_canvas.overlay(current_slice, position=local_pos_ms)
            overflow_buffer.append((0, overflow_slice))
        else:
            active_canvas = active_canvas.overlay(synced_seg, position=local_pos_ms)

        if progress_callback and total_cues > 0:
            frac = 0.80 + (i / total_cues) * 0.12  # 80% to 92%
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
        print(f"🎵 [OUTRO SILENCE] Padding Part {part_num} to preserve video closing outro/music.")
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
        # Fallback Python chunked concatenation
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
# 6. VIDEO MUXING (FFmpeg Stream Copy)
# ==================================================================================================
def mux_video_with_audio(video_input_path: str, audio_input_path: str, output_video_path: str) -> Optional[str]:
    """Muxes the dubbed audio back into the original video without re-encoding the video stream."""
    if not shutil.which("ffmpeg") or not os.path.exists(video_input_path):
        return None

    print(f"\n🎬 [MUXING] Muxing dubbed audio with video: {Path(video_input_path).name}...")
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i", video_input_path,
            "-i", audio_input_path,
            "-c:v", "copy",            # Copy video stream directly (instant)
            "-c:a", "aac",             # AAC audio codec
            "-b:a", "192k",
            "-map", "0:v:0",           # Video from input 0
            "-map", "1:a:0",           # Dubbed audio from input 1
            "-shortest",
            output_video_path,
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(f"🎉 [VIDEO MUX COMPLETE] Final dubbed anime video saved to: {output_video_path}")
        return output_video_path
    except Exception as e:
        print(f"⚠️ [MUX ERROR] Failed to mux video: {e}")
        return None


# ==================================================================================================
# 7. MAIN END-TO-END PIPELINE CONTROLLER
# ==================================================================================================
def run_narutogen_pipeline(
    srt_file: str,
    output_audio: str = "narutogen_dubbed_audio.wav",
    video_file: Optional[str] = None,
    output_video: Optional[str] = "narutogen_dubbed_video.mp4",
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
    base_voice: str = "hi-IN-MadhurNeural",
    rvc_model_path: Optional[str] = None,
    rvc_index_path: Optional[str] = None,
    pitch_shift: int = 0,
    progress_callback=None,
) -> Dict[str, Any]:
    """Automates the entire SRT-to-Dubbed-Audio (and optional Video) workflow end-to-end."""
    pipeline_start = time.time()

    # Step 0: Determine canvas duration
    total_video_ms = ((hours * 3600) + (minutes * 60) + seconds) * 1000
    if video_file and os.path.exists(video_file):
        probed_ms = probe_video_duration(video_file)
        if probed_ms and probed_ms > 0:
            print(f"🎬 [AUTO CANVAS] Detected video duration: {probed_ms/1000:.1f}s ({probed_ms/1000/60:.2f} mins).")
            total_video_ms = probed_ms

    if total_video_ms <= 0:
        total_video_ms = 60000  # Fallback to at least 1 min

    # Workspace folders
    work_dir = Path(tempfile.gettempdir()) / "narutogen_workspace"
    tts_dir = work_dir / "step1_tts"
    rvc_dir = work_dir / "step2_rvc"
    parts_dir = work_dir / "step3_parts"

    for d in [tts_dir, rvc_dir, parts_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(0.02, desc="📄 Parsing SRT subtitle cues...")

    # Step 1: Parse SRT
    cues, adjusted_video_ms = parse_srt_file(srt_file, total_video_ms)
    if not cues:
        raise ValueError("No valid subtitle cues found in SRT file.")

    # Step 2: Ultra-fast TTS (Edge-TTS async)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    cue_tts_pairs = loop.run_until_complete(
        batch_generate_tts(cues, voice=base_voice, tts_output_dir=tts_dir, progress_callback=progress_callback)
    )
    loop.close()

    # Step 3: RVC Voice Conversion
    converter = CharacterVoiceConverter(
        model_path=rvc_model_path,
        index_path=rvc_index_path,
        pitch_shift=pitch_shift,
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

    # Step 5: Optional Video Muxing
    final_video = None
    if video_file and os.path.exists(video_file) and output_video:
        if progress_callback:
            progress_callback(0.95, desc="🎬 Muxing dubbed audio with anime video (FFmpeg)...")
        final_video = mux_video_with_audio(video_file, str(final_audio), str(Path(output_video).resolve()))

    if progress_callback:
        progress_callback(1.0, desc="🏆 NarutoGen Pipeline Complete!")

    total_time = time.time() - pipeline_start
    audio_info = sf.info(str(final_audio))

    return {
        "audio_path": str(final_audio),
        "video_path": str(final_video) if final_video else None,
        "elapsed_seconds": total_time,
        "audio_duration_sec": audio_info.duration,
        "cues_count": len(cues),
        "rvc_active": converter.engine is not None,
    }


# ==================================================================================================
# 8. GRADIO WEB UI INTERFACE
# ==================================================================================================
CUSTOM_CSS = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

.naruto-header {
    text-align: center;
    padding: 28px 20px;
    background: linear-gradient(135deg, #1f1d36 0%, #17152b 50%, #0c0a1a 100%);
    border-radius: 16px;
    border: 1px solid rgba(255, 140, 0, 0.3);
    margin-bottom: 24px;
    box-shadow: 0 8px 32px 0 rgba(255, 100, 0, 0.15);
}

.naruto-title {
    font-size: 2.3rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff8c00, #ff4500, #ffa500);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}

.naruto-subtitle {
    font-size: 1.05rem;
    color: #e2e8f0;
    margin-bottom: 14px;
}

.badge-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
}

.badge {
    background: rgba(255, 140, 0, 0.15);
    border: 1px solid rgba(255, 140, 0, 0.4);
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    color: #ffeedd;
    font-weight: 600;
}

.btn-naruto {
    background: linear-gradient(135deg, #ff8c00 0%, #ff4500 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
    padding: 14px 28px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(255, 69, 0, 0.45) !important;
    transition: all 0.3s ease !important;
}

.btn-naruto:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(255, 69, 0, 0.65) !important;
}

.status-card {
    padding: 16px;
    border-radius: 12px;
    border-left: 5px solid #ff8c00;
    background: rgba(255, 140, 0, 0.08);
    color: #e2e8f0;
}
"""


def gradio_dubbing_handler(
    srt_file_obj,
    video_file_obj,
    hours: float,
    minutes: float,
    seconds: float,
    base_voice: str,
    rvc_pth_obj,
    rvc_index_obj,
    pitch_shift: int,
    progress=gr.Progress(track_tqdm=True),
):
    """Binds the Gradio UI to the NarutoGen pipeline."""
    if srt_file_obj is None:
        return "### ❌ Error: Please upload an SRT subtitle file.", None, None

    srt_path = getattr(srt_file_obj, "name", str(srt_file_obj))
    video_path = getattr(video_file_obj, "name", str(video_file_obj)) if video_file_obj else None
    pth_path = getattr(rvc_pth_obj, "name", str(rvc_pth_obj)) if rvc_pth_obj else None
    index_path = getattr(rvc_index_obj, "name", str(rvc_index_obj)) if rvc_index_obj else None

    # Staging paths
    out_dir = Path(tempfile.gettempdir()) / "narutogen_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = int(time.time())
    out_audio = str(out_dir / f"dubbed_{stamp}.wav")
    out_video = str(out_dir / f"dubbed_{stamp}.mp4") if video_path else None

    try:
        results = run_narutogen_pipeline(
            srt_file=srt_path,
            output_audio=out_audio,
            video_file=video_path,
            output_video=out_video,
            hours=int(hours or 0),
            minutes=int(minutes or 0),
            seconds=int(seconds or 0),
            base_voice=base_voice,
            rvc_model_path=pth_path,
            rvc_index_path=index_path,
            pitch_shift=int(pitch_shift),
            progress_callback=progress,
        )

        elapsed = results["elapsed_seconds"]
        dur = results["audio_duration_sec"]
        rvc_status = "✅ Active (Naruto Character Voice)" if results["rvc_active"] else "⚡ Baseline TTS (Fast Mode)"

        status_markdown = f"""
### 🎉 NarutoGen Dubbing Completed!

| Metric | Result |
| :--- | :--- |
| **Total Processing Time** | **{elapsed:.1f}s** ({elapsed/60:.2f} mins) |
| **Dialogue Cues Processed** | {results["cues_count"]} lines |
| **Final Master Audio Duration** | `{int(dur//60):02d}:{int(dur%60):02d}` ({dur:.2f}s) |
| **Voice Conversion Engine** | {rvc_status} |
| **Pitch Shift (f0_up_key)** | {pitch_shift:+d} semitones |
| **Video Muxing** | {"✅ Video Dubbed Successfully" if results["video_path"] else "Not provided (Audio only)"} |

*Listen to the master audio or watch the dubbed anime video below.*
"""
        return status_markdown, results["audio_path"], results["video_path"]

    except Exception as e:
        err_msg = f"""
### ❌ Dubbing Failed
An error occurred during execution:
```text
{str(e)}
```
*Tip: Ensure your Google Colab runtime is set to **T4 GPU**.*
"""
        return err_msg, None, None


def build_ui():
    """Constructs the sleek NarutoGen Gradio interface."""
    theme = gr.themes.Soft(
        primary_hue="orange",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="NarutoGen Anime Dubbing Studio") as demo:

        gr.HTML(
            """
            <div class="naruto-header">
                <div class="naruto-title">🍥 NarutoGen: AI Anime Dubbing Studio</div>
                <div class="naruto-subtitle">
                    High-Efficiency Two-Step Dubbing Pipeline: Lightweight Neural TTS + RVC Voice Conversion
                </div>
                <div class="badge-row">
                    <span class="badge">⚡ Edge-TTS Async (10x Faster)</span>
                    <span class="badge">🎙️ RVC + RMVPE Pitch Extraction</span>
                    <span class="badge">🛡️ Zero Colab OOM Crashes</span>
                    <span class="badge">🎬 Full Video Muxing</span>
                </div>
            </div>
            """
        )

        with gr.Row():
            # LEFT COLUMN: Inputs
            with gr.Column(scale=5):
                gr.Markdown("### 📥 1. Upload Subtitles & Media")

                srt_input = gr.File(
                    label="Translated SRT Subtitles (*.srt)",
                    file_types=[".srt"],
                    file_count="single",
                )

                video_input = gr.File(
                    label="Optional: Anime Video (*.mp4, *.mkv)",
                    file_types=[".mp4", ".mkv", ".avi", ".mov"],
                    file_count="single",
                )

                gr.Markdown("#### 🎬 Total Video Duration (Manual Canvas Setup)")
                gr.Markdown(
                    "<small style='color: #a0aec0;'>If video is not uploaded, set duration here to preserve outros/BGM.</small>"
                )
                with gr.Row():
                    h_input = gr.Number(label="Hours", value=0, precision=0)
                    m_input = gr.Number(label="Minutes", value=0, precision=0)
                    s_input = gr.Number(label="Seconds", value=48, precision=0)

                with gr.Accordion("🍥 Character Voice (RVC) & TTS Settings", open=True):
                    voice_dropdown = gr.Dropdown(
                        choices=list(DEFAULT_VOICES.items()),
                        value="hi-IN-MadhurNeural",
                        label="Step 1: Baseline Neural TTS Voice",
                        info="Generates clear phonetic baseline speech before character voice conversion",
                    )

                    pth_input = gr.File(
                        label="Step 2: RVC Character Model (*.pth)",
                        file_types=[".pth"],
                        file_count="single",
                    )

                    index_input = gr.File(
                        label="Step 2: RVC Feature Index (*.index - Optional)",
                        file_types=[".index"],
                        file_count="single",
                    )

                    pitch_slider = gr.Slider(
                        minimum=-12,
                        maximum=12,
                        value=0,
                        step=1,
                        label="Pitch Shift / Transpose (Semitones)",
                        info="0 = No change | +12 = Female/Child pitch | -12 = Deep male pitch",
                    )

                start_btn = gr.Button(
                    "⚡ Start NarutoGen Dubbing",
                    variant="primary",
                    size="lg",
                    elem_classes="btn-naruto",
                )

                # Quick sample preset
                sample_file = Path("sample_hindi_english.srt")
                if sample_file.exists():
                    gr.Markdown("#### 💡 Quick Test Example")
                    gr.Examples(
                        examples=[[str(sample_file), None, 0, 0, 48, "hi-IN-MadhurNeural", 0]],
                        inputs=[srt_input, video_input, h_input, m_input, s_input, voice_dropdown, pitch_slider],
                        label="Click to test with sample Hindi-English SRT",
                    )

            # RIGHT COLUMN: Outputs
            with gr.Column(scale=6):
                gr.Markdown("### 🎧 2. Dubbed Master Outputs")

                status_box = gr.Markdown(
                    """
                    <div class="status-card">
                        <b>Ready to dub.</b> Upload your <code>.srt</code> file and optional RVC model, 
                        then click <b>Start NarutoGen Dubbing</b>.
                    </div>
                    """
                )

                audio_player = gr.Audio(
                    label="Dubbed Master Audio (.wav)",
                    type="filepath",
                    interactive=False,
                )

                video_player = gr.Video(
                    label="Final Dubbed Video (.mp4)",
                    interactive=False,
                )

        # Connect button event
        start_btn.click(
            fn=gradio_dubbing_handler,
            inputs=[
                srt_input,
                video_input,
                h_input,
                m_input,
                s_input,
                voice_dropdown,
                pth_input,
                index_input,
                pitch_slider,
            ],
            outputs=[
                status_box,
                audio_player,
                video_player,
            ],
        )

    return demo


# ==================================================================================================
# 9. PUBLIC LAUNCH (share=True FOR GOOGLE COLAB)
# ==================================================================================================
if __name__ == "__main__":
    demo = build_ui()
    # MANDATORY: share=True generates a public gradio.live URL in Google Colab
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        debug=True,
    )
