"""
====================================================================================================
NARUTOGEN MASTER PIPELINE RUNNER
====================================================================================================
Integrates:
- Module 1: Edge-TTS Base Speech Generation (Asynchronous & Free)
- Module 2: RVC Batch Voice Conversion (Naruto .pth & .index with RMVPE)
- Module 3: Audio Stitching & Precise Time-Synchronization (FFmpeg/Pydub)
====================================================================================================
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

# Ensure UTF-8 console output on Windows
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from modules.tts_generator import generate_base_speech, DEFAULT_VOICE, SUPPORTED_VOICES
from modules.rvc_converter import run_rvc_conversion
from modules.audio_stitcher import stitch_audio_chunks


def run_narutogen_pipeline(
    input_file: str,
    output_master_wav: str = "./outputs/naruto_master.wav",
    model_path: Optional[str] = "./models/naruto/naruto.pth",
    index_path: Optional[str] = "./models/naruto/naruto.index",
    base_voice: str = DEFAULT_VOICE,
    pitch_shift: int = 0,
    f0_method: str = "rmvpe",
    hours: int = 0,
    minutes: int = 0,
    seconds: int = 0,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Executes the end-to-end NarutoGen dubbing pipeline:
    Input (.srt / .txt) -> Base TTS (.wav) -> RVC Naruto (.wav) -> Stitched Master (.wav)
    """
    pipeline_start = time.time()
    print("\n" + "=" * 70)
    print("NARUTOGEN AUTOMATED AUDIO DUBBING PIPELINE")
    print("=" * 70)
    print(f"Input File:        {input_file}")
    print(f"Base Neural Voice: {base_voice}")
    print(f"Character Model:   {model_path if model_path and os.path.exists(model_path) else 'Base TTS Mode (No .pth)'}")
    print(f"Pitch Transpose:   {pitch_shift:+d} semitones | Algorithm: {f0_method}")
    print(f"Master Audio Dest: {output_master_wav}")
    print("=" * 70 + "\n")

    # Step 0: Total duration calculation
    total_canvas_ms = ((hours * 3600) + (minutes * 60) + seconds) * 1000 if (hours or minutes or seconds) else None

    # Step 1: Base TTS Generation via Edge-TTS
    t0 = time.time()
    print("[PHASE 1/3] Generating Baseline Audio via Microsoft Edge-TTS...")
    cues = generate_base_speech(
        input_source=input_file,
        output_dir="./outputs/temp_base_tts",
        voice=base_voice,
        concurrency=10,
        progress_callback=progress_callback,
    )
    t1 = time.time()
    print(f"Phase 1 Complete in {t1 - t0:.2f} seconds ({len(cues)} dialogue chunks generated).\n")

    # Step 2: RVC Character Voice Conversion
    t2 = time.time()
    if model_path and os.path.exists(model_path):
        print("[PHASE 2/3] Transforming Timbre to Naruto via RVC (RMVPE)...")
        cues = run_rvc_conversion(
            input_items=cues,
            model_path=model_path,
            index_path=index_path,
            output_dir="./outputs/temp_rvc_converted",
            pitch_shift=pitch_shift,
            f0_method=f0_method,
            progress_callback=progress_callback,
        )
    else:
        print("[PHASE 2/3] Skipping RVC (Model .pth not provided). Using Baseline Hindi Audio.")
    t3 = time.time()
    print(f"Phase 2 Complete in {t3 - t2:.2f} seconds.\n")

    # Step 3: Exact Time-Synchronization & Timeline Stitching
    t4 = time.time()
    print("[PHASE 3/3] Synchronizing Dialogue & Stitching Master Timeline Canvas...")
    final_master_path = stitch_audio_chunks(
        cues=cues,
        output_master_wav=output_master_wav,
        total_duration_ms=total_canvas_ms,
        progress_callback=progress_callback,
    )
    t5 = time.time()
    print(f"Phase 3 Complete in {t5 - t4:.2f} seconds.\n")

    total_time = time.time() - pipeline_start
    print("=" * 70)
    print("NARUTOGEN PIPELINE FINISHED SUCCESSFULLY!")
    print(f"Total Execution Time: {total_time:.2f} seconds ({total_time / 60:.2f} mins)")
    print(f"Master Audio File:    {Path(final_master_path).resolve()}")
    print("=" * 70 + "\n")

    return {
        "master_wav": str(final_master_path),
        "total_cues": len(cues),
        "elapsed_seconds": total_time,
    }


def main():
    parser = argparse.ArgumentParser(
        description="NarutoGen: Automated AI Anime Dubbing & Voice Conversion Pipeline"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="inputs/sample_hindi_english.srt",
        help="Path to input .srt subtitle file or text script",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="outputs/naruto_master.wav",
        help="Path for final master output .wav file",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="models/naruto/naruto.pth",
        help="Path to pre-downloaded Naruto RVC .pth weights",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="models/naruto/naruto.index",
        help="Path to Naruto RVC feature .index file",
    )
    parser.add_argument(
        "--voice", "-v",
        type=str,
        default=DEFAULT_VOICE,
        help=f"Edge-TTS base voice name. Supported presets: {list(SUPPORTED_VOICES.keys())}",
    )
    parser.add_argument(
        "--pitch", "-p",
        type=int,
        default=0,
        help="Pitch transpose in semitones (0 for natural, +12 for octave up, -12 down)",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=0,
        help="Total video duration - Hours (to preserve outros/BGM)",
    )
    parser.add_argument(
        "--minutes",
        type=int,
        default=0,
        help="Total video duration - Minutes",
    )
    parser.add_argument(
        "--seconds",
        type=int,
        default=0,
        help="Total video duration - Seconds",
    )

    args = parser.parse_args()

    run_narutogen_pipeline(
        input_file=args.input,
        output_master_wav=args.output,
        model_path=args.model if os.path.exists(args.model) else None,
        index_path=args.index if os.path.exists(args.index) else None,
        base_voice=args.voice,
        pitch_shift=args.pitch,
        hours=args.hours,
        minutes=args.minutes,
        seconds=args.seconds,
    )


if __name__ == "__main__":
    main()