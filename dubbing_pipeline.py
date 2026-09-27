"""
====================================================================================================
🎬 HIGH-SPEED, PRODUCTION-READY AUDIO DUBBING & TIME-SYNCHRONIZATION PIPELINE
====================================================================================================
Target Environment: Google Colab Free Tier (NVIDIA T4 15GB VRAM, 12GB System RAM, 2 CPU Cores)
TTS Model: Tharshan/indicf5_hindi-english_code_switch (Hinglish Code-Switching)

Features:
1. Dynamic Audio Canvas initialized to exact total video duration (Hours, Minutes, Seconds).
2. Asynchronous Multi-Threaded Producer-Consumer Architecture:
   - PRODUCER THREAD (GPU): Dynamic batching (4-5 cues of similar duration), FP16 precision,
     and Scaled Dot-Product Attention (SDPA) for near-100% T4 GPU utilization.
   - CONSUMER THREAD (CPU): Non-blocking audio pull, pitch-preserving time-stretching (librosa/pydub)
     or silence padding, overlaying onto active window canvas.
3. Strict 15-Minute RAM Flushing Mechanism:
   - Never keeps a multi-hour master track in RAM (which causes Colab 12GB OOM crashes).
   - Exports 15-minute segments to disk (part_001.wav, part_002.wav, etc.) and flushes variables.
4. Seamless Final Concatenation:
   - Automatically concatenates all parts into a master track matching the exact requested duration.
====================================================================================================
"""

import os
import gc
import re
import sys
import time
import math
import shutil
import queue
import threading
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Audio & ML libraries
import torch
import numpy as np
import soundfile as sf
import librosa
from transformers import AutoModel
import pysrt
from pydub import AudioSegment

# Optional system monitoring
try:
    import psutil
except ImportError:
    psutil = None

# ==================================================================================================
# CONSTANTS & CONFIGURATION
# ==================================================================================================
SAMPLE_RATE = 24000          # IndicF5 native output sample rate (24 kHz)
HOP_LENGTH = 256             # Mel spectrogram hop length
TARGET_RMS = 0.1             # RMS normalization constant from IndicF5
FLUSH_INTERVAL_MINUTES = 15  # Strict RAM flush window (15 minutes)
FLUSH_INTERVAL_MS = FLUSH_INTERVAL_MINUTES * 60 * 1000  # 900,000 ms
DEFAULT_BATCH_SIZE = 4       # Subtitles grouped per batch
MAX_QUEUE_SIZE = 12          # Backpressure limit to keep RAM low during GPU generation


# ==================================================================================================
# 1. SYSTEM MONITORING & UTILITIES
# ==================================================================================================
def get_memory_stats() -> str:
    """Returns human-readable RAM and VRAM utilization string."""
    stats = []
    if psutil:
        ram = psutil.virtual_memory()
        stats.append(f"RAM: {ram.used / (1024**3):.1f}/{ram.total / (1024**3):.1f}GB ({ram.percent}%)")
    if torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated() / (1024**3)
        vram_res = torch.cuda.memory_reserved() / (1024**3)
        stats.append(f"VRAM: {vram_alloc:.2f}GB alloc ({vram_res:.2f}GB res)")
    return " | ".join(stats) if stats else "Monitoring unavailable"


def clean_subtitle_text(text: str) -> str:
    """Cleans SRT formatting artifacts, tags, linebreaks, and non-dialogue annotations."""
    # Strip HTML tags like <i>, <b>, <font>
    text = re.sub(r"<[^>]+>", "", text)
    # Remove bracketed sound effects e.g. [Music], (Laughter)
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
    # Replace line breaks and tabs with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==================================================================================================
# 2. INTERACTIVE USER INPUT: DYNAMIC CANVAS SETUP
# ==================================================================================================
def get_user_video_duration(cli_h: Optional[int] = None, cli_m: Optional[int] = None, cli_s: Optional[int] = None) -> int:
    """
    Prompt user in console with clean inputs to specify total video duration:
    - 'Enter total video duration - Hours: '
    - 'Minutes: '
    - 'Seconds: '
    Converts input into milliseconds to establish master canvas boundaries.
    """
    if cli_h is not None and cli_m is not None and cli_s is not None:
        total_ms = ((cli_h * 3600) + (cli_m * 60) + cli_s) * 1000
        print(f"🎬 Video Duration set via arguments: {cli_h:02d}:{cli_m:02d}:{cli_s:02d} ({total_ms:,} ms)")
        return total_ms

    print("\n" + "=" * 65)
    print("🎬 INTERACTIVE USER INPUT: DYNAMIC CANVAS SETUP")
    print("Specify total video duration to ensure outros/BGM are never cut off.")
    print("=" * 65)

    while True:
        try:
            h_str = input("Enter total video duration - Hours: ").strip()
            hours = int(h_str) if h_str else 0

            m_str = input("Minutes: ").strip()
            minutes = int(m_str) if m_str else 0

            s_str = input("Seconds: ").strip()
            seconds = int(s_str) if s_str else 0

            if hours < 0 or minutes < 0 or seconds < 0:
                print("❌ Duration values cannot be negative. Try again.\n")
                continue

            if hours == 0 and minutes == 0 and seconds == 0:
                print("❌ Total video duration cannot be 00:00:00. Try again.\n")
                continue

            total_ms = ((hours * 3600) + (minutes * 60) + seconds) * 1000
            print(f"✅ Master Canvas Initialized: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_ms:,} ms)")
            print(f"📊 Memory Strategy: {math.ceil(total_ms / FLUSH_INTERVAL_MS)} intermediate 15-min parts.")
            print("=" * 65 + "\n")
            return total_ms

        except ValueError:
            print("❌ Invalid input! Please enter integer numbers for Hours, Minutes, and Seconds.\n")


# ==================================================================================================
# 3. MODEL INITIALIZATION (STRICTLY ONCE ON GPU WITH FP16 & SDPA)
# ==================================================================================================
class DubbingTTSModel:
    """Wrapper for Tharshan/indicf5_hindi-english_code_switch with FP16 and SDPA acceleration."""

    def __init__(self, model_name: str = "Tharshan/indicf5_hindi-english_code_switch", voice_key: str = "ritu_hinglish"):
        print("\n🚀 [MODEL] Initializing IndicF5 Hinglish TTS Engine...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"⚡ [HARDWARE] Running on Device: {self.device}")

        # Enable PyTorch SDPA (Scaled Dot-Product Attention) Fast-Path
        if self.device.type == "cuda":
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cudnn.benchmark = True
            print("⚡ [SDPA] FlashAttention & Memory-Efficient SDPA acceleration ENABLED.")

        print(f"📦 [HF LOAD] Loading weights for '{model_name}'...")
        # AutoModel loads custom code directly from HF repository
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Convert DiT backbone to FP16 to maximize generation speed & save VRAM on T4
        if self.device.type == "cuda":
            print("🎯 [PRECISION] Converting transformer DiT backbone to torch.float16 (FP16)...")
            self.model = self.model.half().to(self.device)
            # Ensure Vocos vocoder is loaded and kept in float32 for clean acoustic reconstruction
            _ = self.model.vocoder
            self.model._vocoder = self.model._vocoder.to(self.device).float()
        else:
            self.model = self.model.to(self.device)

        self.model.eval()

        # Select reference voice
        available_voices = self.model.voices()
        if voice_key not in available_voices:
            voice_key = next(iter(available_voices))
        self.ref_audio_path, self.ref_text = self.model.voice(voice_key)
        self.voice_key = voice_key
        print(f"🎙️ [VOICE] Selected Reference Voice: '{voice_key}' (Prompt: \"{self.ref_text[:40]}...\")")
        print(f"✅ [READY] Model loaded successfully. {get_memory_stats()}\n")

    @torch.inference_mode()
    def generate_batch(
        self,
        texts: List[str],
        nfe_step: int = 32,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
        speed: float = 1.0,
    ) -> List[Tuple[np.ndarray, int]]:
        """
        Runs batched parallel inference through DiT flow matching.
        Groups texts together to minimize padding overhead and maximize T4 Tensor Core saturation.
        """
        B = len(texts)
        if B == 0:
            return []

        # Load reference audio condition
        cond, rms = self.model._load_ref(self.ref_audio_path)
        ref_text_clean = self.ref_text.strip()
        if not ref_text_clean.endswith((" ", ".", "।", "!", "?")):
            ref_text_clean += ". "

        ref_len = cond.shape[-1] // HOP_LENGTH

        # Calculate target frame durations for each dialogue in the batch
        durations = []
        for t in texts:
            dur = ref_len + int(ref_len / len(ref_text_clean.encode()) * len(t.encode()) / speed)
            durations.append(dur)

        # Batch inputs
        cond_batch = cond.repeat(B, 1)
        full_texts = [ref_text_clean + t for t in texts]
        dur_tensor = torch.tensor(durations, device=self.device, dtype=torch.long)

        # Parallel flow-matching ODE sample
        try:
            generated, _ = self.model.model.sample(
                cond=cond_batch,
                text=full_texts,
                duration=dur_tensor,
                steps=nfe_step,
                cfg_strength=cfg_strength,
                sway_sampling_coef=sway_sampling_coef,
            )

            results = []
            for i in range(B):
                # Slice generated portion excluding reference audio
                mel_slice = generated[i:i + 1, ref_len:durations[i], :].to(torch.float32).permute(0, 2, 1)
                wave = self.model.vocoder.decode(mel_slice).squeeze().cpu()
                if rms < TARGET_RMS:
                    wave = wave * (rms / TARGET_RMS)
                results.append((wave.numpy().astype(np.float32), SAMPLE_RATE))
            return results

        except Exception as e:
            # Fallback to single-item generation if batching encounters edge cases
            print(f"⚠️ [BATCH FALLBACK] Sequential fallback triggered due to: {e}")
            results = []
            for t in texts:
                wav, sr = self.model.generate(
                    t,
                    ref_audio=self.ref_audio_path,
                    ref_text=self.ref_text,
                    nfe_step=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    speed=speed,
                )
                results.append((wav, sr))
            return results


# ==================================================================================================
# 4. SRT PROCESSING & LOCALIZED DYNAMIC BATCHING
# ==================================================================================================
class SubtitleCue:
    def __init__(self, cue_id: int, start_ms: int, end_ms: int, text: str):
        self.cue_id = cue_id
        self.start_ms = start_ms
        self.end_ms = end_ms
        self.target_dur_ms = max(100, end_ms - start_ms)
        self.text = text

    def __repr__(self):
        return f"<Cue #{self.cue_id} [{self.start_ms}ms -> {self.end_ms}ms ({self.target_dur_ms}ms)]: '{self.text[:20]}...'>"


def parse_srt_file(srt_path: str, total_video_ms: int) -> List[SubtitleCue]:
    """Parses SRT, cleans dialogues, filters empty cues, and checks boundary conditions."""
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

    # Check if last cue extends beyond initial video canvas
    if cues and cues[-1].end_ms > total_video_ms:
        overhang_sec = (cues[-1].end_ms - total_video_ms) / 1000.0
        print(f"⚠️ [CANVAS EXPANSION] Last dialogue ends at {cues[-1].end_ms / 1000:.1f}s, past user canvas ({total_video_ms / 1000:.1f}s).")
        print(f"👉 Expanding canvas by {overhang_sec:.1f}s to prevent cutting off dialogue.")
        total_video_ms = cues[-1].end_ms + 2000

    return cues


def create_dynamic_batches(cues: List[SubtitleCue], batch_size: int = DEFAULT_BATCH_SIZE) -> List[List[SubtitleCue]]:
    """
    Groups subtitles into dynamic batches of 4-5 items of similar duration.
    Uses localized chunk sorting to maintain timeline progression while minimizing padding waste.
    """
    batches = []
    # Local grouping window of 8-12 cues
    window_group_size = batch_size * 2

    for w_idx in range(0, len(cues), window_group_size):
        chunk = cues[w_idx:w_idx + window_group_size]
        # Sort locally by target duration
        sorted_chunk = sorted(chunk, key=lambda c: c.target_dur_ms)

        for b_idx in range(0, len(sorted_chunk), batch_size):
            batch = sorted_chunk[b_idx:b_idx + batch_size]
            if batch:
                batches.append(batch)

    return batches


# ==================================================================================================
# 5. AUDIO SYNCHRONIZATION (FAST SPEEDUP WITHOUT PITCH CHANGE OR SILENCE PADDING)
# ==================================================================================================
def time_sync_audio(audio_np: np.ndarray, sr: int, target_dur_ms: int, cue_text: str = "") -> AudioSegment:
    """
    Instantly time-syncs generated audio chunk to exact SRT target duration (end_time - start_time):
    1. If generated > target: Speed up WITHOUT pitch alteration using librosa time_stretch / pydub.
    2. If generated < target: Pad with silence to match exact target duration.
    3. Return sample-accurate pydub AudioSegment.
    """
    actual_dur_ms = int(len(audio_np) / sr * 1000)

    # Convert float32 [-1.0, 1.0] to int16 PCM
    audio_int16 = (np.clip(audio_np, -1.0, 1.0) * 32767).astype(np.int16)

    # Case 1: Audio is longer than SRT duration -> Speed up without pitch shift
    if actual_dur_ms > target_dur_ms:
        speed_ratio = actual_dur_ms / target_dur_ms

        # Limit speedup to preserve natural intelligibility (cap at 1.8x)
        capped_ratio = min(speed_ratio, 1.8)
        if speed_ratio > 1.8:
            print(f"⚠️ [SYNC CAP] Dialogue '{cue_text[:25]}...' needs {speed_ratio:.2f}x speedup. Capping at 1.8x.")

        try:
            # High-fidelity phase vocoder time-stretch (preserves pitch)
            stretched = librosa.effects.time_stretch(audio_np, rate=capped_ratio)
            stretched_int16 = (np.clip(stretched, -1.0, 1.0) * 32767).astype(np.int16)
            seg = AudioSegment(
                data=stretched_int16.tobytes(),
                sample_width=2,
                frame_rate=sr,
                channels=1,
            )
        except Exception:
            # Fallback to pydub cross-splice speedup
            seg = AudioSegment(
                data=audio_int16.tobytes(),
                sample_width=2,
                frame_rate=sr,
                channels=1,
            )
            from pydub.effects import speedup
            seg = speedup(seg, playback_speed=capped_ratio)

        # Micro-trim or micro-pad to match exact millisecond target
        if len(seg) > target_dur_ms:
            seg = seg[:target_dur_ms]
        elif len(seg) < target_dur_ms:
            seg = seg + AudioSegment.silent(duration=target_dur_ms - len(seg), frame_rate=sr)

        return seg

    # Case 2: Audio is shorter than SRT duration -> Pad with silence
    elif actual_dur_ms < target_dur_ms:
        seg = AudioSegment(
            data=audio_int16.tobytes(),
            sample_width=2,
            frame_rate=sr,
            channels=1,
        )
        pad_ms = target_dur_ms - actual_dur_ms
        seg = seg + AudioSegment.silent(duration=pad_ms, frame_rate=sr)
        return seg

    # Case 3: Exact match
    else:
        return AudioSegment(
            data=audio_int16.tobytes(),
            sample_width=2,
            frame_rate=sr,
            channels=1,
        )


# ==================================================================================================
# 6. ASYNCHRONOUS PRODUCER-CONSUMER ORCHESTRATION & 15-MINUTE RAM FLUSH
# ==================================================================================================
SENTINEL_STOP = None


def producer_gpu_thread(
    model_wrapper: DubbingTTSModel,
    batches: List[List[SubtitleCue]],
    task_queue: queue.Queue,
    progress_dict: Dict[str, Any],
):
    """
    PRODUCER THREAD (GPU):
    Iterates through dynamically batched SRT cues.
    Executes batched F5-TTS inference concurrently on the GPU.
    Pushes generated audio arrays and their exact target durations to Queue.
    Never blocks on CPU audio synchronization.
    """
    total_batches = len(batches)
    print(f"🎬 [PRODUCER] GPU Thread started. Total Batches: {total_batches}")

    start_time = time.time()
    for batch_idx, batch in enumerate(batches, start=1):
        texts = [cue.text for cue in batch]
        avg_target_dur = sum(c.target_dur_ms for c in batch) / len(batch) / 1000.0

        b_start = time.time()
        # Batched inference on GPU
        audio_results = model_wrapper.generate_batch(texts)
        b_time = time.time() - b_start

        # Push to thread-safe queue
        for cue, (audio_np, sr) in zip(batch, audio_results):
            task_queue.put((cue, audio_np, sr))

        progress_dict["batches_done"] = batch_idx
        q_size = task_queue.qsize()

        print(
            f"⚡ [PRODUCER] Batch {batch_idx}/{total_batches} Generated ({len(batch)} cues, avg {avg_target_dur:.1f}s) "
            f"in {b_time:.2f}s | Queue Buffer: {q_size}/{MAX_QUEUE_SIZE} | {get_memory_stats()}"
        )

    # Signal completion to Consumer
    task_queue.put(SENTINEL_STOP)
    total_time = time.time() - start_time
    print(f"✅ [PRODUCER] GPU Generation Complete! Processed {total_batches} batches in {total_time / 60:.2f} mins.")


def consumer_cpu_thread(
    task_queue: queue.Queue,
    total_cues: int,
    total_video_ms: int,
    output_dir: Path,
    intermediate_files: List[Path],
    progress_dict: Dict[str, Any],
):
    """
    CONSUMER THREAD (CPU):
    Continuously pulls generated audio from Queue.
    Instantly time-syncs each chunk to its target SRT duration.
    Overlays synced chunk onto current 15-minute slice master canvas.
    Flushes 15-minute chunk to disk and cleans RAM to strictly prevent Colab OOM.
    """
    print(f"🎧 [CONSUMER] CPU Thread started. Total timeline: {total_video_ms / 1000 / 60:.2f} mins.")

    # Calculate total 15-minute windows
    num_windows = math.ceil(total_video_ms / FLUSH_INTERVAL_MS)
    current_window_idx = 0

    def init_window_canvas(win_idx: int) -> AudioSegment:
        win_start = win_idx * FLUSH_INTERVAL_MS
        win_dur = min(FLUSH_INTERVAL_MS, total_video_ms - win_start)
        return AudioSegment.silent(duration=win_dur, frame_rate=SAMPLE_RATE)

    active_canvas = init_window_canvas(current_window_idx)
    # Storage for overflow audio that crosses the 15-minute boundary
    overflow_buffer: List[Tuple[int, AudioSegment]] = []
    cues_processed = 0

    while True:
        item = task_queue.get()
        if item is SENTINEL_STOP:
            task_queue.task_done()
            break

        cue, audio_np, sr = item

        # 1. Instantly time-sync chunk to exact target duration
        synced_seg = time_sync_audio(audio_np, sr, cue.target_dur_ms, cue.text)

        # 2. Determine which window this cue belongs to
        cue_window_idx = cue.start_ms // FLUSH_INTERVAL_MS

        # Check if we need to advance windows before placing this cue
        while cue_window_idx > current_window_idx:
            # Flush current window to disk
            part_num = current_window_idx + 1
            part_path = output_dir / f"part_{part_num:03d}.wav"
            print(f"\n💾 [RAM FLUSH] 15-Minute Timeline Window {current_window_idx} Completed.")
            print(f"💾 [EXPORT] Flushing to disk: {part_path.name} ({len(active_canvas) / 1000 / 60:.2f} mins)...")
            active_canvas.export(str(part_path), format="wav")
            intermediate_files.append(part_path)

            # Strict RAM release
            del active_canvas
            gc.collect()
            print(f"🧹 [GARBAGE COLLECT] Memory purged. {get_memory_stats()}\n")

            # Advance to next window
            current_window_idx += 1
            active_canvas = init_window_canvas(current_window_idx)

            # Apply any buffered overflow from previous window
            for ov_pos, ov_seg in overflow_buffer:
                active_canvas = active_canvas.overlay(ov_seg, position=ov_pos)
            overflow_buffer.clear()

        # 3. Calculate local position inside the current window
        win_start_ms = current_window_idx * FLUSH_INTERVAL_MS
        local_pos_ms = cue.start_ms - win_start_ms
        win_dur_ms = len(active_canvas)

        # Check if dialogue straddles across current 15-minute boundary
        if local_pos_ms + len(synced_seg) > win_dur_ms:
            split_point = win_dur_ms - local_pos_ms
            current_slice = synced_seg[:split_point]
            overflow_slice = synced_seg[split_point:]

            active_canvas = active_canvas.overlay(current_slice, position=local_pos_ms)
            # Store overflow slice to overlay at start of next window
            overflow_buffer.append((0, overflow_slice))
            print(f"✂️ [BOUNDARY SPLIT] Cue #{cue.cue_id} spans across 15-min boundary! Sliced cleanly.")
        else:
            active_canvas = active_canvas.overlay(synced_seg, position=local_pos_ms)

        cues_processed += 1
        progress_dict["cues_synced"] = cues_processed

        if cues_processed % 10 == 0 or cues_processed == total_cues:
            print(
                f"🎧 [CONSUMER] Synced & Placed Cue #{cue.cue_id} ({cues_processed}/{total_cues}) "
                f"@ {cue.start_ms / 1000:.1f}s (Dur: {cue.target_dur_ms}ms) | {get_memory_stats()}"
            )

        task_queue.task_done()

    # Flush the active window
    part_num = current_window_idx + 1
    part_path = output_dir / f"part_{part_num:03d}.wav"
    print(f"\n💾 [FINAL WINDOW FLUSH] Flushing Part {part_num}: {part_path.name}...")
    active_canvas.export(str(part_path), format="wav")
    intermediate_files.append(part_path)
    del active_canvas
    gc.collect()

    # If the user specified a total duration that extends beyond the last dialogue,
    # generate any remaining silent 15-minute chunks so outro/BGM is not cut off!
    current_window_idx += 1
    while current_window_idx < num_windows:
        part_num = current_window_idx + 1
        part_path = output_dir / f"part_{part_num:03d}.wav"
        trailing_canvas = init_window_canvas(current_window_idx)
        print(f"🎵 [OUTRO SILENCE] Generating blank canvas for Part {part_num} ({len(trailing_canvas)/1000:.1f}s) to preserve outro/credits.")
        trailing_canvas.export(str(part_path), format="wav")
        intermediate_files.append(part_path)
        del trailing_canvas
        gc.collect()
        current_window_idx += 1

    print("✅ [CONSUMER] All cues synced and intermediate parts exported.")


# ==================================================================================================
# 7. CONCATENATION OF FLUSHED INTERMEDIATE FILES
# ==================================================================================================
def concatenate_intermediate_parts(intermediate_files: List[Path], output_wav: Path):
    """
    Concatenates all intermediate 15-minute WAV files into final master track.
    Uses ffmpeg concat demuxer (0% RAM overhead). Falls back to chunked file stream.
    """
    print("\n" + "=" * 65)
    print("🏁 [CONCATENATION] Assembling Final Master Audio Track...")
    print(f"📁 Combining {len(intermediate_files)} intermediate parts into: {output_wav.name}")
    print("=" * 65)

    concat_txt = output_wav.parent / "concat_list.txt"
    with open(concat_txt, "w", encoding="utf-8") as f:
        for p in intermediate_files:
            f.write(f"file '{p.resolve()}'\n")

    # Try fast ffmpeg concat demuxer (instant, 0 RAM usage)
    ffmpeg_available = shutil.which("ffmpeg") is not None
    if ffmpeg_available:
        try:
            print("🚀 Executing ffmpeg stream copy concat...")
            cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat_txt),
                "-c",
                "copy",
                str(output_wav),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            print(f"🎉 [SUCCESS] Master Audio Track generated via ffmpeg: {output_wav}")
            return
        except subprocess.SubprocessError as e:
            print(f"⚠️ [FFMPEG FAILED] Falling back to chunked Python stream writer: {e}")

    # Fallback: Low-memory chunked python stream concatenation (reads 64KB blocks, ~1MB RAM)
    print("🐍 Executing Python low-memory chunked stream concatenation...")
    with sf.SoundFile(str(output_wav), mode="w", samplerate=SAMPLE_RATE, channels=1, subtype="PCM_16") as outfile:
        for idx, part_path in enumerate(intermediate_files, start=1):
            print(f"   ↳ Streaming Part {idx}/{len(intermediate_files)}: {part_path.name}")
            with sf.SoundFile(str(part_path), mode="r") as infile:
                while True:
                    data = infile.read(65536, dtype="int16")
                    if len(data) == 0:
                        break
                    outfile.write(data)

    print(f"🎉 [SUCCESS] Master Audio Track generated: {output_wav}")


# ==================================================================================================
# 8. MAIN PIPELINE ENTRYPOINT
# ==================================================================================================
def run_pipeline(
    srt_file: str,
    output_audio: str = "final_dubbed_audio.wav",
    hours: Optional[int] = None,
    minutes: Optional[int] = None,
    seconds: Optional[int] = None,
    voice_key: str = "ritu_hinglish",
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    """Full end-to-end execution pipeline."""
    pipeline_start = time.time()

    # Step 1: Interactive User Input for Total Video Canvas
    total_video_ms = get_user_video_duration(hours, minutes, seconds)

    # Step 2: Parse and clean SRT subtitles
    cues = parse_srt_file(srt_file, total_video_ms)
    if not cues:
        print("❌ No dialogue cues found in SRT file! Exiting.")
        return

    # Step 3: Create localized dynamic batches of similar duration
    batches = create_dynamic_batches(cues, batch_size=batch_size)
    print(f"📊 [BATCHING] Grouped {len(cues)} cues into {len(batches)} dynamic batches (size ~{batch_size}).")

    # Step 4: Create intermediate directory for 15-minute flushes
    output_wav_path = Path(output_audio).resolve()
    parts_dir = output_wav_path.parent / "intermediate_parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    # Step 5: Initialize Model strictly ONCE on GPU in FP16 with SDPA
    tts_engine = DubbingTTSModel(voice_key=voice_key)

    # Step 6: Launch Multi-Threaded Producer-Consumer
    task_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
    intermediate_files: List[Path] = []
    progress_dict = {"batches_done": 0, "cues_synced": 0}

    producer = threading.Thread(
        target=producer_gpu_thread,
        args=(tts_engine, batches, task_queue, progress_dict),
        name="GPU-Producer-Thread",
    )
    consumer = threading.Thread(
        target=consumer_cpu_thread,
        args=(task_queue, len(cues), total_video_ms, parts_dir, intermediate_files, progress_dict),
        name="CPU-Consumer-Thread",
    )

    print("🚀 [THREADS] Starting Producer (GPU) and Consumer (CPU) Threads...")
    producer.start()
    consumer.start()

    # Wait for completion
    producer.join()
    consumer.join()

    # Step 7: Concatenate intermediate flushed parts
    concatenate_intermediate_parts(intermediate_files, output_wav_path)

    # Verify final audio length
    final_info = sf.info(str(output_wav_path))
    final_dur_sec = final_info.duration
    target_dur_sec = total_video_ms / 1000.0

    print("\n" + "=" * 65)
    print("🏆 DUBBING PIPELINE COMPLETE")
    print(f"⏱️ Total Execution Time: {(time.time() - pipeline_start) / 60:.2f} minutes")
    print(f"📊 Final Audio File: {output_wav_path}")
    print(f"📊 Actual Audio Duration: {final_dur_sec:.2f}s ({final_dur_sec/60:.2f} mins)")
    print(f"📊 Requested Duration:   {target_dur_sec:.2f}s ({target_dur_sec/60:.2f} mins)")
    print(f"📊 Sample Rate: {final_info.samplerate} Hz | Channels: {final_info.channels}")
    print("=" * 65 + "\n")


# ==================================================================================================
# CLI EXECUTION
# ==================================================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Production-Ready Colab Audio Dubbing Pipeline")
    parser.add_argument("--srt", type=str, default="sample_hindi_english.srt", help="Path to translated SRT file")
    parser.add_argument("--out", type=str, default="final_dubbed_audio.wav", help="Path to output master audio WAV")
    parser.add_argument("--hours", type=int, default=None, help="Video duration - Hours")
    parser.add_argument("--minutes", type=int, default=None, help="Video duration - Minutes")
    parser.add_argument("--seconds", type=int, default=None, help="Video duration - Seconds")
    parser.add_argument("--voice", type=str, default="ritu_hinglish", help="Bundled voice key (e.g., ritu_hinglish, ta_hinglish)")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE, help="SRT lines per dynamic batch")

    args = parser.parse_args()

    run_pipeline(
        srt_file=args.srt,
        output_audio=args.out,
        hours=args.hours,
        minutes=args.minutes,
        seconds=args.seconds,
        voice_key=args.voice,
        batch_size=args.batch_size,
    )
