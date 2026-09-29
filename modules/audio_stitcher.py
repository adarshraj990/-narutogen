"""
====================================================================================================
MODULE 3: AUDIO STITCHER & TIMELINE SYNCHRONIZER
====================================================================================================
Stitches and time-aligns all converted character audio chunks into a continuous master audio file.
Features:
- Pitch-Preserving Time-Stretch: High-fidelity phase vocoder (librosa/pydub) ensures dialogue fits
  the exact SRT timestamp window without pitch distortion (no chipmunk effect).
- Master Timeline Canvas: Accurately places each dialogue at its designated start timestamp.
- Outro & BGM Preservation: Allows setting total video duration so closing music is not truncated.
- 15-Minute RAM Flushing: Slices and exports audio in 15-minute segments to disk, preventing
  system RAM bloat and OOM crashes on long-form (up to 2-3 hours) content.
- Zero-RAM Concatenation: Uses FFmpeg concat demuxer (stream copy) to merge all parts in seconds.
====================================================================================================
"""

import os
import gc
import math
import shutil
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

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
    SOUNDFILE_AVAILABLE = False

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

from .tts_generator import SubtitleCue

SAMPLE_RATE = 44100
FLUSH_INTERVAL_MINUTES = 15
FLUSH_INTERVAL_MS = FLUSH_INTERVAL_MINUTES * 60 * 1000  # 900,000 ms


class AudioStitcher:
    """
    Handles precise time synchronization and master timeline stitching.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, max_speedup: float = 1.75):
        self.sample_rate = sample_rate
        self.max_speedup = max_speedup

    def sync_dialogue_chunk(self, audio_path: str, target_dur_ms: int) -> AudioSegment:
        """
        Synchronizes an individual audio chunk to exact target duration (end_time - start_time):
        1. If audio is longer than timestamp window:
           Applies pitch-preserving time stretch (phase vocoder / pydub speedup).
        2. If audio is shorter than timestamp window:
           Pads with silence to reach target duration.
        """
        if not os.path.exists(audio_path):
            # Fallback to silence
            return AudioSegment.silent(duration=target_dur_ms, frame_rate=self.sample_rate)

        # Method 1: Librosa phase-vocoder time-stretch (highest quality)
        if LIBROSA_AVAILABLE:
            try:
                y, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
                actual_dur_ms = int(len(y) / sr * 1000)

                if actual_dur_ms > target_dur_ms:
                    speed_ratio = actual_dur_ms / target_dur_ms
                    capped_ratio = min(speed_ratio, self.max_speedup)

                    stretched = librosa.effects.time_stretch(y, rate=capped_ratio)
                    stretched_int16 = (np.clip(stretched, -1.0, 1.0) * 32767).astype(np.int16)
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

            except Exception as e:
                print(f"⚠️ [STITCHER WARNING] Phase-vocoder stretch error: {e}")

        # Method 2: Pydub fallback
        try:
            seg = AudioSegment.from_file(audio_path)
            actual_dur_ms = len(seg)
            if actual_dur_ms > target_dur_ms:
                speed_ratio = actual_dur_ms / target_dur_ms
                try:
                    from pydub.effects import speedup
                    seg = speedup(seg, playback_speed=min(speed_ratio, self.max_speedup))
                except Exception:
                    pass
                if len(seg) > target_dur_ms:
                    seg = seg[:target_dur_ms]
                return seg
            elif actual_dur_ms < target_dur_ms:
                pad_ms = target_dur_ms - actual_dur_ms
                return seg + AudioSegment.silent(duration=pad_ms, frame_rate=self.sample_rate)
            return seg
        except Exception as e:
            return AudioSegment.silent(duration=target_dur_ms, frame_rate=self.sample_rate)

    def stitch_timeline(
        self,
        cues: List[SubtitleCue],
        output_master_wav: str = "./outputs/master_dubbed_audio.wav",
        total_duration_ms: Optional[int] = None,
        scratch_dir: str = "./outputs/stitch_parts",
        progress_callback=None,
    ) -> str:
        """
        Assembles all cues onto the master timeline using a 15-minute slice-and-flush strategy.
        Guarantees zero memory overflow for long-form content.
        """
        if not cues:
            raise ValueError("No subtitle cues provided for timeline stitching.")

        out_master_path = Path(output_master_wav).resolve()
        out_master_path.parent.mkdir(parents=True, exist_ok=True)

        parts_dir = Path(scratch_dir).resolve()
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        parts_dir.mkdir(parents=True, exist_ok=True)

        # Determine canvas total duration
        max_cue_end = max(c.end_ms for c in cues)
        if total_duration_ms is None or total_duration_ms < max_cue_end:
            canvas_total_ms = max_cue_end + 2000  # 2s trailing padding
        else:
            canvas_total_ms = total_duration_ms

        num_windows = math.ceil(canvas_total_ms / FLUSH_INTERVAL_MS)
        current_window_idx = 0
        intermediate_files: List[Path] = []

        print("\n" + "=" * 65)
        print("🎧 [STITCHER] Assembling Master Audio Timeline...")
        print(f"⏱️ Total Timeline: {canvas_total_ms / 1000 / 60:.2f} mins ({canvas_total_ms:,} ms)")
        print(f"🛡️ Memory Strategy: {num_windows} slice(s) of 15-minute RAM flushes")
        print(f"📦 Total Dialogue Chunks: {len(cues)}")
        print("=" * 65)

        def init_window_canvas(win_idx: int) -> AudioSegment:
            win_start = win_idx * FLUSH_INTERVAL_MS
            win_dur = min(FLUSH_INTERVAL_MS, canvas_total_ms - win_start)
            return AudioSegment.silent(duration=win_dur, frame_rate=self.sample_rate)

        active_canvas = init_window_canvas(current_window_idx)
        overflow_buffer: List[Tuple[int, AudioSegment]] = []
        total_cues = len(cues)

        for i, cue in enumerate(cues, start=1):
            audio_source = cue.rvc_wav_path if cue.rvc_wav_path else cue.base_wav_path
            synced_seg = self.sync_dialogue_chunk(audio_source, cue.target_dur_ms)

            cue_window_idx = cue.start_ms // FLUSH_INTERVAL_MS

            # Flush current window to disk if the cue starts in a future 15-min window
            while cue_window_idx > current_window_idx:
                part_num = current_window_idx + 1
                part_path = parts_dir / f"part_{part_num:03d}.wav"
                print(f"💾 [FLUSH] Exporting 15-min Part {part_num} to disk ({part_path.name})...")
                active_canvas.export(str(part_path), format="wav")
                intermediate_files.append(part_path)

                # Purge RAM
                del active_canvas
                gc.collect()

                # Initialize next window
                current_window_idx += 1
                active_canvas = init_window_canvas(current_window_idx)

                # Apply overflow from previous window
                for ov_pos, ov_seg in overflow_buffer:
                    active_canvas = active_canvas.overlay(ov_seg, position=ov_pos)
                overflow_buffer.clear()

            # Local position inside active 15-minute window
            win_start_ms = current_window_idx * FLUSH_INTERVAL_MS
            local_pos_ms = cue.start_ms - win_start_ms
            win_dur_ms = len(active_canvas)

            # Check if dialogue straddles across 15-min window boundary
            if local_pos_ms + len(synced_seg) > win_dur_ms:
                split_point = win_dur_ms - local_pos_ms
                current_slice = synced_seg[:split_point]
                overflow_slice = synced_seg[split_point:]

                active_canvas = active_canvas.overlay(current_slice, position=local_pos_ms)
                overflow_buffer.append((0, overflow_slice))
            else:
                active_canvas = active_canvas.overlay(synced_seg, position=local_pos_ms)

            if progress_callback and total_cues > 0:
                progress_callback(i / total_cues, desc=f"Stitching dialogue: {i}/{total_cues} cues...")

        # Flush final active window
        part_num = current_window_idx + 1
        part_path = parts_dir / f"part_{part_num:03d}.wav"
        print(f"💾 [FLUSH] Exporting final Part {part_num} ({part_path.name})...")
        active_canvas.export(str(part_path), format="wav")
        intermediate_files.append(part_path)
        del active_canvas
        gc.collect()

        # Outro padding: If total duration extends past final dialogue, export trailing silence
        current_window_idx += 1
        while current_window_idx < num_windows:
            part_num = current_window_idx + 1
            part_path = parts_dir / f"part_{part_num:03d}.wav"
            trailing_canvas = init_window_canvas(current_window_idx)
            print(f"🎵 [OUTRO SILENCE] Exporting trailing silence Part {part_num} to preserve outro/BGM.")
            trailing_canvas.export(str(part_path), format="wav")
            intermediate_files.append(part_path)
            del trailing_canvas
            gc.collect()
            current_window_idx += 1

        # Concatenate intermediate parts into the final master WAV
        self._concatenate_parts(intermediate_files, out_master_path, parts_dir)
        return str(out_master_path)

    def _concatenate_parts(self, intermediate_files: List[Path], out_master_path: Path, parts_dir: Path):
        """Concatenates flushed intermediate WAV parts via FFmpeg concat demuxer."""
        concat_txt = parts_dir / "concat_list.txt"
        with open(concat_txt, "w", encoding="utf-8") as f:
            for p in intermediate_files:
                f.write(f"file '{p.resolve()}'\n")

        ffmpeg_cmd = shutil.which("ffmpeg")
        if ffmpeg_cmd:
            print("🚀 [CONCAT] Concatenating parts via FFmpeg stream copy (0 MB RAM overhead)...")
            cmd = [
                ffmpeg_cmd,
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_txt),
                "-c", "copy",
                str(out_master_path),
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        else:
            # Fallback Python chunked file stream (low memory ~1MB RAM)
            print("🐍 [CONCAT] FFmpeg not found. Using low-memory chunked Python stream concatenation...")
            with sf.SoundFile(str(out_master_path), mode="w", samplerate=self.sample_rate, channels=1, subtype="PCM_16") as outfile:
                for p in intermediate_files:
                    with sf.SoundFile(str(p), mode="r") as infile:
                        while True:
                            data = infile.read(65536, dtype="int16")
                            if len(data) == 0:
                                break
                            outfile.write(data)

        # Validate master file
        info = sf.info(str(out_master_path))
        print(f"🎉 [MASTER AUDIO CREATED] {out_master_path.name}")
        print(f"📊 Duration: {info.duration:.2f}s ({info.duration/60:.2f} mins) | Sample Rate: {info.samplerate} Hz")


# ==================================================================================================
# MODULE ENTRYPOINT FUNCTION
# ==================================================================================================
def stitch_audio_chunks(
    cues: List[SubtitleCue],
    output_master_wav: str = "./outputs/master_dubbed_audio.wav",
    total_duration_ms: Optional[int] = None,
    progress_callback=None,
) -> str:
    """
    High-level entrypoint for Module 3.
    """
    stitcher = AudioStitcher()
    return stitcher.stitch_timeline(
        cues=cues,
        output_master_wav=output_master_wav,
        total_duration_ms=total_duration_ms,
        progress_callback=progress_callback,
    )
