# 🍥 NarutoGen: Automated Local & Colab AI Audio Conversion Pipeline
### High-Speed Anime Voice Conversion (Edge-TTS + RVC RMVPE + Precision Stitching)
Optimized for Anti-Gravity IDE & Google Colab Free Tier (T4 GPU, 16GB RAM) | 100% Free, NO Paid APIs, Zero OOM Crashes.

---

## 📁 Project Directory Structure

```text
NARUTOGEN/
├── models/                     # Character RVC V2 weights (.pth & .index)
│   └── naruto/
│       ├── README.md           # Instructions on placing Naruto RVC weights
│       ├── naruto.pth          # (Place pre-downloaded Naruto weights here)
│       └── naruto.index        # (Place pre-downloaded feature index here)
├── modules/                    # Clean, modular processing components
│   ├── __init__.py             # Module package exports
│   ├── tts_generator.py        # Module 1: Edge-TTS synthesis (SRT & plain text)
│   ├── rvc_converter.py        # Module 2: RVC batch voice conversion (RMVPE)
│   └── audio_stitcher.py       # Module 3: Time-sync, canvas assembly & FFmpeg stitching
├── inputs/                     # Input scripts & subtitle files
│   └── sample_hindi_english.srt
├── outputs/                    # Exported continuous master audio (.wav)
│   └── .gitkeep
├── pipeline.py                 # Automated master CLI pipeline runner
├── dubbing_pipeline.py         # Full Gradio Web UI application (gr.Blocks + share=True)
└── requirements.txt            # Python dependencies
```

---

## 🧩 Modular Components Breakdown

### 1. Module 1: TTS Generation (`modules/tts_generator.py`)
- **Technology:** Microsoft `edge-tts` (asynchronous websocket synthesis).
- **Zero API Keys:** 100% free, runs without account setup or billing.
- **Speed:** Synthesizes ~1,200 dialogue lines in **1–2 minutes** using async concurrency (`asyncio.Semaphore`).
- **Multi-Language Voices:**
  - Hindi Male: `hi-IN-MadhurNeural` (heroic, clear, assertive — ideal base for Naruto)
  - Hindi Female: `hi-IN-SwaraNeural` (female characters / young Naruto)
  - Spanish: `es-ES-AlvaroNeural`, `es-MX-JorgeNeural`
  - French: `fr-FR-HenriNeural`
  - English / Hinglish: `en-US-GuyNeural`, `en-IN-PrabhatNeural`
- **Output:** Clean 44.1kHz 16-bit PCM `.wav` chunks.

### 2. Module 2: RVC Voice Conversion (`modules/rvc_converter.py`)
- **Technology:** Retrieval-based Voice Conversion (RVC V2).
- **Pitch Algorithm:** `rmvpe` (RMVPE provides superior polyphonic pitch estimation, eliminating metallic/robotic artifacts).
- **Persistent GPU Cache:** Loads model onto GPU memory strictly once, processing all dialogue chunks in batches without re-initializing the neural net.
- **Parameters:**
  - `pitch_shift` (`f0_up_key`): `-12` to `+12` semitones.
  - `index_rate`: `0.75` (retrieval feature strength).
  - `protect`: `0.33` (protects unvoiced consonants).

### 3. Module 3: Audio Stitching & Synchronization (`modules/audio_stitcher.py`)
- **Pitch-Preserving Time-Stretch:** Uses `librosa` phase-vocoder (or `pydub` cross-splice) to automatically speed up dialogue chunks that exceed their subtitle timestamp window without altering pitch.
- **Silence Padding:** Automatically pads shorter lines with silence so dialogue stays in exact lip-sync.
- **15-Minute RAM Flushing:** Slices and exports audio in 15-minute segments to disk, purging memory variables (`gc.collect()`) so 2-hour audio never crashes Colab's 12GB RAM.
- **Zero-RAM Concatenation:** Merges all 15-minute segments into the final master `.wav` file via FFmpeg stream copy (or low-memory Python chunk streaming).

---

## ⚡ Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
# On Linux / Colab:
apt-get install -y ffmpeg
```

### 2. Run Master CLI Pipeline
```bash
# Basic run with sample SRT:
python pipeline.py --input inputs/sample_hindi_english.srt --output outputs/naruto_master.wav

# Run with custom Naruto RVC model:
python pipeline.py \
  --input inputs/sample_hindi_english.srt \
  --model models/naruto/naruto.pth \
  --index models/naruto/naruto.index \
  --voice hi_madhur \
  --pitch 0 \
  --output outputs/naruto_master.wav
```

### 3. Run Gradio Web UI (with Colab share=True public link)
```bash
python dubbing_pipeline.py
```
Open the generated `https://xxxx.gradio.live` link in any browser to drag-and-drop SRT files, select character models, and monitor real-time progress.
