"""
====================================================================================================
🎙️ HIGH-SPEED AUDIO DUBBING & TIME-SYNC PIPELINE (GRADIO WEB APP)
====================================================================================================
Target Environment: Google Colab Free Tier (NVIDIA T4 15GB VRAM, 12GB System RAM, 2 CPU Cores)
TTS Model: Tharshan/indicf5_hindi-english_code_switch (Hinglish Code-Switching)

Key Capabilities:
1. Modern Gradio Web UI (gr.Blocks) with instant public link (share=True) for Google Colab.
2. Zero Terminal Blocking: No input() statements; all parameters entered via sleek web interface.
3. Asynchronous Producer-Consumer Threading:
   - PRODUCER (GPU): Dynamic batching (4-5 cues of similar duration), FP16 precision,
     and Scaled Dot-Product Attention (SDPA) for high T4 GPU utilization.
   - CONSUMER (CPU): Non-blocking queue pull, pitch-preserving time-stretching (librosa/pydub)
     or silence padding, overlaying onto active window canvas.
4. Strict 15-Minute RAM Flushing:
   - Prevents Colab 12GB OOM crashes by flushing audio every 15 minutes of timeline to disk
     (part_001.wav, part_002.wav, etc.) and explicitly purging memory variables.
5. Master Canvas Duration Matching:
   - Automatically pads silence up to the user-specified video duration so intros/outros/BGM
     are never cut off.
6. Seamless Final Concatenation:
   - Assembles all parts via ffmpeg stream copy (0 MB RAM overhead) into a complete master audio file.
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
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

# Core ML & Audio Processing
import torch
import numpy as np
import soundfile as sf
import librosa
from transformers import AutoModel
import pysrt
from pydub import AudioSegment
import gradio as gr

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

# Global cache for the loaded model to prevent reloading on multiple runs
CACHED_MODEL_WRAPPER = None


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
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\[.*?\]|\(.*?\)", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ==================================================================================================
# 2. MODEL INITIALIZATION (STRICTLY ONCE ON GPU WITH FP16 & SDPA)
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
        self.set_voice(voice_key)
        print(f"✅ [READY] Model loaded successfully. {get_memory_stats()}\n")

    def set_voice(self, voice_key: str):
        available_voices = self.model.voices()
        if voice_key not in available_voices:
            voice_key = next(iter(available_voices))
        self.ref_audio_path, self.ref_text = self.model.voice(voice_key)
        self.voice_key = voice_key
        print(f"🎙️ [VOICE] Selected Reference Voice: '{voice_key}' (Prompt: \"{self.ref_text[:35]}...\")")

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
            print(f"⚠️ [BATCH FALLBACK] Sequential fallback triggered: {e}")
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


def get_tts_engine(voice_key: str = "ritu_hinglish") -> DubbingTTSModel:
    """Returns cached model instance or initializes it once."""
    global CACHED_MODEL_WRAPPER
    if CACHED_MODEL_WRAPPER is None:
        CACHED_MODEL_WRAPPER = DubbingTTSModel(voice_key=voice_key)
    else:
        CACHED_MODEL_WRAPPER.set_voice(voice_key)
    return CACHED_MODEL_WRAPPER


# ==================================================================================================
# 3. SRT PROCESSING & LOCALIZED DYNAMIC BATCHING
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


def parse_srt_file(srt_path: str, total_video_ms: int) -> Tuple[List[SubtitleCue], int]:
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
    adjusted_video_ms = total_video_ms
    if cues and cues[-1].end_ms > total_video_ms:
        overhang_sec = (cues[-1].end_ms - total_video_ms) / 1000.0
        print(f"⚠️ [CANVAS EXPANSION] Last dialogue ends at {cues[-1].end_ms / 1000:.1f}s, past user canvas ({total_video_ms / 1000:.1f}s).")
        print(f"👉 Expanding canvas by {overhang_sec:.1f}s to prevent cutting off dialogue.")
        adjusted_video_ms = cues[-1].end_ms + 2000

    return cues, adjusted_video_ms


def create_dynamic_batches(cues: List[SubtitleCue], batch_size: int = DEFAULT_BATCH_SIZE) -> List[List[SubtitleCue]]:
    """
    Groups subtitles into dynamic batches of 4-5 items of similar duration.
    Uses localized chunk sorting to maintain timeline progression while minimizing padding waste.
    """
    batches = []
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
# 4. AUDIO SYNCHRONIZATION (FAST SPEEDUP WITHOUT PITCH CHANGE OR SILENCE PADDING)
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
        capped_ratio = min(speed_ratio, 1.8)

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
# 5. ASYNCHRONOUS PRODUCER-CONSUMER ORCHESTRATION & 15-MINUTE RAM FLUSH
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

    num_windows = math.ceil(total_video_ms / FLUSH_INTERVAL_MS)
    current_window_idx = 0

    def init_window_canvas(win_idx: int) -> AudioSegment:
        win_start = win_idx * FLUSH_INTERVAL_MS
        win_dur = min(FLUSH_INTERVAL_MS, total_video_ms - win_start)
        return AudioSegment.silent(duration=win_dur, frame_rate=SAMPLE_RATE)

    active_canvas = init_window_canvas(current_window_idx)
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

    # Outro preservation: Generate remaining silent chunks if video is longer than dialogues
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
# 6. CONCATENATION OF FLUSHED INTERMEDIATE FILES
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

    # Fast ffmpeg concat demuxer (instantaneous stream copy, 0 RAM overhead)
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
# 7. CORE PIPELINE CONTROLLER (NON-BLOCKING)
# ==================================================================================================
def execute_dubbing_pipeline(
    srt_file_path: str,
    output_audio_path: str,
    total_video_ms: int,
    voice_key: str = "ritu_hinglish",
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_callback=None,
) -> str:
    """
    Non-blocking pipeline execution function called directly by the Gradio web handler.
    Does not use terminal input(); all parameters are supplied programmatically.
    """
    pipeline_start = time.time()

    if progress_callback:
        progress_callback(0.05, desc="Parsing SRT subtitles...")

    # Step 1: Parse and clean SRT subtitles
    cues, adjusted_video_ms = parse_srt_file(srt_file_path, total_video_ms)
    if not cues:
        raise ValueError("No valid dialogue cues found in the uploaded SRT file.")

    # Step 2: Create localized dynamic batches of similar duration
    batches = create_dynamic_batches(cues, batch_size=batch_size)
    print(f"📊 [BATCHING] Grouped {len(cues)} cues into {len(batches)} dynamic batches (size ~{batch_size}).")

    # Step 3: Create intermediate directory for 15-minute flushes
    output_wav_path = Path(output_audio_path).resolve()
    parts_dir = output_wav_path.parent / "intermediate_parts"
    if parts_dir.exists():
        shutil.rmtree(parts_dir)
    parts_dir.mkdir(parents=True, exist_ok=True)

    if progress_callback:
        progress_callback(0.15, desc="Initializing IndicF5 Hinglish Model (FP16 & SDPA)...")

    # Step 4: Initialize or retrieve cached Model on GPU
    tts_engine = get_tts_engine(voice_key=voice_key)

    if progress_callback:
        progress_callback(0.25, desc="Starting Asynchronous GPU Producer & CPU Consumer Threads...")

    # Step 5: Launch Multi-Threaded Producer-Consumer
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
        args=(task_queue, len(cues), adjusted_video_ms, parts_dir, intermediate_files, progress_dict),
        name="CPU-Consumer-Thread",
    )

    producer.start()
    consumer.start()

    # Track thread execution with UI progress bar updates
    total_batches = len(batches)
    while producer.is_alive() or consumer.is_alive():
        if progress_callback and total_batches > 0:
            done = progress_dict.get("batches_done", 0)
            frac = min(0.90, 0.25 + (done / total_batches) * 0.65)
            progress_callback(frac, desc=f"Generating Batch {done}/{total_batches} on T4 GPU...")
        time.sleep(0.5)

    producer.join()
    consumer.join()

    if progress_callback:
        progress_callback(0.92, desc="Concatenating flushed 15-minute WAV files...")

    # Step 6: Concatenate intermediate flushed parts
    concatenate_intermediate_parts(intermediate_files, output_wav_path)

    if progress_callback:
        progress_callback(1.0, desc="Dubbing Pipeline Complete!")

    # Verify final audio length
    final_info = sf.info(str(output_wav_path))
    print(f"\n🏆 Master Audio Generated: {output_wav_path} ({final_info.duration:.2f}s in {(time.time() - pipeline_start)/60:.2f} mins)")
    return str(output_wav_path)


# Backward-compatible alias
run_pipeline = execute_dubbing_pipeline


# ==================================================================================================
# 8. GRADIO WEB UI INTERFACE (gr.Blocks)
# ==================================================================================================
CUSTOM_CSS = """
/* Container & typography */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header styling */
.header-box {
    text-align: center;
    padding: 26px 20px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 22px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff7e5f, #feb47b, #7f7fd5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 6px;
}

.header-subtitle {
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
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    color: #f7fafc;
    font-weight: 500;
}

/* Primary Action Button */
.action-btn {
    background: linear-gradient(135deg, #ff5e3a 0%, #ff2a6d 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.15rem !important;
    padding: 14px 28px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(255, 42, 109, 0.4) !important;
    transition: all 0.3s ease !important;
}

.action-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(255, 42, 109, 0.6) !important;
}

/* Status cards */
.status-box {
    padding: 16px;
    border-radius: 12px;
    border-left: 5px solid #00f2fe;
    background: rgba(0, 242, 254, 0.05);
    color: #e2e8f0;
}
"""


def gradio_dubbing_handler(
    srt_file_obj,
    hours: float,
    minutes: float,
    seconds: float,
    voice_key: str,
    batch_size: int,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Connects the Gradio frontend to the asynchronous dubbing backend:
    - Strictly non-blocking: parses UI inputs and triggers engine.
    - Yields real-time status and returns audio file for playback/download.
    """
    # Validation 1: SRT Upload
    if srt_file_obj is None:
        err_markdown = (
            "### ❌ Error: Missing Subtitle File\n"
            "Please upload a translated `.srt` subtitle file to begin dubbing."
        )
        return err_markdown, None

    srt_path = getattr(srt_file_obj, "name", str(srt_file_obj))
    if not os.path.exists(srt_path):
        err_markdown = "### ❌ Error: The uploaded file was not found on disk. Please re-upload."
        return err_markdown, None

    # Validation 2: Video Duration
    h = int(hours or 0)
    m = int(minutes or 0)
    s = int(seconds or 0)

    if h < 0 or m < 0 or s < 0:
        err_markdown = "### ❌ Error: Duration values cannot be negative."
        return err_markdown, None

    if h == 0 and m == 0 and s == 0:
        err_markdown = (
            "### ❌ Error: Total Video Duration cannot be 00:00:00\n"
            "Please enter the video's total duration (Hours, Minutes, Seconds) to ensure closing credits "
            "and outro music are properly preserved in the master audio canvas."
        )
        return err_markdown, None

    total_video_ms = ((h * 3600) + (m * 60) + s) * 1000

    # Staging output directory
    output_dir = Path(tempfile.gettempdir()) / "indicf5_dubbing_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    output_wav_path = str(output_dir / f"dubbed_audio_{timestamp}.wav")

    start_clock = time.time()
    try:
        final_wav = execute_dubbing_pipeline(
            srt_file_path=srt_path,
            output_audio_path=output_wav_path,
            total_video_ms=total_video_ms,
            voice_key=voice_key,
            batch_size=int(batch_size),
            progress_callback=progress,
        )

        elapsed = time.time() - start_clock
        info = sf.info(final_wav)
        actual_dur_sec = info.duration
        file_size_mb = os.path.getsize(final_wav) / (1024 * 1024)

        status_markdown = f"""
### 🎉 Dubbing Pipeline Completed Successfully!

| Metric | Details |
| :--- | :--- |
| **Status** | ✅ Generated in **{elapsed:.1f}s** ({elapsed/60:.2f} mins) |
| **Requested Canvas Duration** | `{h:02d}:{m:02d}:{s:02d}` ({total_video_ms // 1000}s) |
| **Master Audio Duration** | `{int(actual_dur_sec//60):02d}:{int(actual_dur_sec%60):02d}` ({actual_dur_sec:.2f}s) |
| **Audio Format** | {info.samplerate} Hz (PCM 16-bit Mono) |
| **File Size** | {file_size_mb:.2f} MB |
| **Voice Profile** | `{voice_key}` |
| **Colab Safety** | Flushed in 15-minute segments (0 OOM) |

*You can now play the audio track or download the WAV file using the player below.*
"""
        return status_markdown, final_wav

    except Exception as e:
        err_markdown = f"""
### ❌ Pipeline Execution Failed
An error occurred during audio dubbing:
```text
{str(e)}
```
*Tip: Ensure your Google Colab runtime is set to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).*
"""
        return err_markdown, None


def build_ui():
    """Builds the modern Gradio Blocks UI layout."""
    theme = gr.themes.Soft(
        primary_hue="rose",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="IndicF5 Audio Dubbing Studio") as demo:

        # Hero Banner
        gr.HTML(
            """
            <div class="header-box">
                <div class="header-title">🎙️ IndicF5 Hinglish Dubbing Studio</div>
                <div class="header-subtitle">
                    High-Speed, Production-Ready Audio Dubbing & Exact Time-Synchronization Pipeline
                </div>
                <div class="badge-row">
                    <span class="badge">⚡ T4 GPU Optimized (FP16 + SDPA)</span>
                    <span class="badge">🧠 Dynamic Batching (4-5 Cues)</span>
                    <span class="badge">🛡️ 15-Min RAM Flushing Safe</span>
                    <span class="badge">🎵 Zero-Pitch Distortion Sync</span>
                </div>
            </div>
            """
        )

        with gr.Row():
            # LEFT COLUMN: Inputs
            with gr.Column(scale=5):
                gr.Markdown("### 📥 1. Upload Subtitles & Canvas Setup")

                srt_file_input = gr.File(
                    label="Upload Translated Subtitles (*.srt)",
                    file_types=[".srt"],
                    file_count="single",
                )

                gr.Markdown("#### 🎬 Total Video Duration")
                gr.Markdown(
                    "<small style='color: #a0aec0;'>Enter the exact total video duration. This guarantees that "
                    "background music and closing credits after the final dialogue are not cut off.</small>"
                )

                with gr.Row():
                    hours_input = gr.Number(
                        label="Hours",
                        value=0,
                        minimum=0,
                        step=1,
                        precision=0,
                    )
                    minutes_input = gr.Number(
                        label="Minutes",
                        value=0,
                        minimum=0,
                        maximum=59,
                        step=1,
                        precision=0,
                    )
                    seconds_input = gr.Number(
                        label="Seconds",
                        value=48,
                        minimum=0,
                        maximum=59,
                        step=1,
                        precision=0,
                    )

                with gr.Accordion("⚙️ Advanced Voice & GPU Options", open=False):
                    voice_dropdown = gr.Dropdown(
                        choices=[
                            ("Ritu (Hinglish Female - Default)", "ritu_hinglish"),
                            ("Tamil-Hinglish", "ta_hinglish"),
                            ("Bengali", "bn"),
                            ("Gujarati", "gu"),
                            ("Kannada", "kn"),
                            ("Malayalam", "ml"),
                            ("Marathi", "mr"),
                            ("Odia", "or"),
                            ("Punjabi", "pa"),
                            ("Telugu", "te"),
                        ],
                        value="ritu_hinglish",
                        label="Reference Voice Profile",
                        info="Bundled Indic reference voice for speech synthesis",
                    )
                    batch_slider = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=DEFAULT_BATCH_SIZE,
                        step=1,
                        label="Dynamic Batch Size (Cues per Batch)",
                        info="Groups similar-duration lines to maximize T4 Tensor Core speed",
                    )

                start_btn = gr.Button(
                    "⚡ Start Dubbing Pipeline",
                    variant="primary",
                    size="lg",
                    elem_classes="action-btn",
                )

                # Quick Example Card
                sample_file = Path("sample_hindi_english.srt")
                if sample_file.exists():
                    gr.Markdown("#### 💡 Quick Test Example")
                    gr.Examples(
                        examples=[[str(sample_file), 0, 0, 48, "ritu_hinglish", 4]],
                        inputs=[
                            srt_file_input,
                            hours_input,
                            minutes_input,
                            seconds_input,
                            voice_dropdown,
                            batch_slider,
                        ],
                        label="Click to load sample Hindi-English code-switched SRT",
                    )

            # RIGHT COLUMN: Status & Output Audio
            with gr.Column(scale=6):
                gr.Markdown("### 🎧 2. Real-Time Status & Audio Player")

                status_box = gr.Markdown(
                    """
                    <div class="status-box">
                        <b>Ready to process.</b> Upload your <code>.srt</code> file and set video duration, 
                        then click <b>Start Dubbing Pipeline</b>.
                    </div>
                    """
                )

                audio_player = gr.Audio(
                    label="Generated Dubbed Master Audio (.wav)",
                    type="filepath",
                    interactive=False,
                )

                gr.Markdown(
                    """
                    > **🛡️ Colab Free Tier RAM Safeguard:**  
                    > For long 2–3 hour videos, audio is automatically processed and exported in **15-minute flushed chunks**, 
                    > keeping system RAM strictly under Colab's 12GB ceiling without crashing.
                    """
                )

        # Wire event listener
        start_btn.click(
            fn=gradio_dubbing_handler,
            inputs=[
                srt_file_input,
                hours_input,
                minutes_input,
                seconds_input,
                voice_dropdown,
                batch_slider,
            ],
            outputs=[
                status_box,
                audio_player,
            ],
        )

    return demo


# ==================================================================================================
# 9. PUBLIC LAUNCH WITH share=True FOR GOOGLE COLAB
# ==================================================================================================
if __name__ == "__main__":
    demo = build_ui()
    # MANDATORY: share=True generates a public gradio.live URL in Google Colab terminal
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        debug=True,
    )
