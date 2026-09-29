"""
====================================================================================================
TEST & DEMONSTRATION: NARUTO RVC MULTILINGUAL VOICE CONVERSION
====================================================================================================
Verifies end-to-end integration:
1. Automatic download & directory verification of the Naruto RVC V2 model (.pth and .index).
2. Base audio generation with Microsoft Edge-TTS across 4 languages:
   - Hindi (hi-IN-MadhurNeural)
   - Spanish (es-ES-AlvaroNeural)
   - French (fr-FR-HenriNeural)
   - Portuguese (pt-BR-AntonioNeural)
3. Batch voice conversion through RVC with RMVPE pitch extraction into Naruto's voice.
4. Export and validation of final converted .wav files in ./outputs/naruto_rvc/
====================================================================================================
"""

import os
import sys
import time
from pathlib import Path

if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import soundfile as sf

from modules.model_downloader import ensure_naruto_model
from modules.tts_generator import generate_base_speech, SUPPORTED_VOICES, resolve_voice_name
from modules.rvc_converter import convert_base_audio_to_naruto, RVCBatchConverter

SAMPLE_DIALOGUES = [
    {
        "lang": "hindi",
        "voice": "hi-IN-MadhurNeural",
        "tag": "naruto_hindi",
        "text": "मेरा नाम नारुतो उज़ुमाकी है, और मैं गाँव का सबसे महान होकेगे बनूँगा! दात्तेबायो!",
    },
    {
        "lang": "spanish",
        "voice": "es-ES-AlvaroNeural",
        "tag": "naruto_spanish",
        "text": "¡Mi nombre es Naruto Uzumaki y seré el próximo Hokage de la Aldea de la Hoja!",
    },
    {
        "lang": "french",
        "voice": "fr-FR-HenriNeural",
        "tag": "naruto_french",
        "text": "Je m'appelle Naruto Uzumaki et je ne reviens jamais sur ma parole!",
    },
    {
        "lang": "portuguese",
        "voice": "pt-BR-AntonioNeural",
        "tag": "naruto_portuguese",
        "text": "Meu nome é Naruto Uzumaki e eu vou me tornar o maior Hokage de todos!",
    },
]


def run_multilingual_naruto_test():
    total_start = time.time()
    print("\n" + "=" * 75)
    print("🍥 NARUTO RVC MULTILINGUAL CONVERSION TEST (Hindi, Spanish, French, Portuguese)")
    print("=" * 75)

    # 1. Check & Ensure Naruto RVC Model
    print("\n[STEP 1] Checking Naruto RVC Model weights and index...")
    model_info = ensure_naruto_model()
    model_path = model_info["model_path"]
    index_path = model_info["index_path"]
    print(f"   Model Weights: {model_path}")
    print(f"   Feature Index: {index_path}")

    # 2. Generate Base Audio with Edge-TTS for all 4 languages
    print("\n[STEP 2] Generating Base Audio via Microsoft Edge-TTS for 4 Languages...")
    base_audio_dir = Path("./outputs/temp_multilingual_base").resolve()
    base_audio_dir.mkdir(parents=True, exist_ok=True)
    generated_base_files = []

    for item in SAMPLE_DIALOGUES:
        tag = item["tag"]
        voice = item["voice"]
        text = item["text"]
        lang = item["lang"]

        print(f"   🎤 Synthesizing {lang.capitalize()} ({voice})...")
        cues = generate_base_speech(
            input_source=text,
            output_dir=str(base_audio_dir / tag),
            voice=voice,
        )
        if cues and cues[0].base_wav_path and os.path.exists(cues[0].base_wav_path):
            base_file = cues[0].base_wav_path
            # Copy to distinct name
            dest_file = base_audio_dir / f"{tag}_base.wav"
            import shutil
            shutil.copy2(base_file, dest_file)
            generated_base_files.append(str(dest_file))
            info = sf.info(str(dest_file))
            print(f"      ↳ Base audio created: {dest_file.name} ({info.duration:.2f}s, {info.samplerate}Hz)")

    # 3. Batch Convert Base Audio to Naruto Voice via RVC
    print("\n[STEP 3] Converting Base Audio to Naruto's Voice via RVC...")
    rvc_out_dir = Path("./outputs/naruto_rvc").resolve()
    rvc_out_dir.mkdir(parents=True, exist_ok=True)

    converted_files = convert_base_audio_to_naruto(
        audio_inputs=generated_base_files,
        output_dir=str(rvc_out_dir),
        pitch_shift=0,
        f0_method="rmvpe",
        model_path=model_path,
        index_path=index_path,
    )

    # 4. Verify Final Audio Outputs
    print("\n[STEP 4] Verifying Final Converted Audio Files:")
    print("=" * 75)
    for f in converted_files:
        p = Path(f)
        if p.exists() and p.stat().st_size > 0:
            info = sf.info(str(p))
            size_mb = p.stat().st_size / (1024 * 1024)
            print(f"   ✅ [OK] {p.name}")
            print(f"          Duration:    {info.duration:.2f} seconds")
            print(f"          Sample Rate: {info.samplerate} Hz (PCM {info.subtype})")
            print(f"          File Size:   {size_mb:.2f} MB")
            print(f"          Path:        {p}\n")
        else:
            print(f"   ❌ [FAILED] {p.name} - File not created or empty!")

    elapsed = time.time() - total_start
    print("=" * 75)
    print(f"🎉 All {len(converted_files)} multilingual audio files generated in {elapsed:.2f} seconds!")
    print(f"📁 Designated Output Directory: {rvc_out_dir}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_multilingual_naruto_test()
