# 🍥 NarutoGen: High-Efficiency AI Anime Dubbing Pipeline
### Two-Step Architecture (Lightweight Neural TTS + RVC Voice Conversion)
Optimized for Google Colab Free Tier (T4 GPU, 16GB RAM) | 2-Hour Video Dubbed in **10–15 Minutes** (Zero OOM Crashes)

---

## 🚀 Why This Architecture?

| Approach | 2-Hour Execution Time | Memory / Stability | Verdict |
| :--- | :--- | :--- | :--- |
| **Heavy Diffusion / Flow-Matching (e.g. IndicF5)** | > 70–90 Minutes ❌ | Constant Colab 12GB OOM Crashes ❌ | **Rejected** |
| **NarutoGen Two-Step (Edge-TTS + RVC RMVPE)** | **10–15 Minutes** ⚡ | Zero OOM (Under 2GB RAM & 3GB VRAM) ✅ | **Adopted ✅** |

1. **Step 1 (Ultra-Fast Synthesis):** Asynchronous Microsoft Edge Neural TTS generates standard Hindi/Hinglish baseline audio for all 1,200+ subtitles in **~1–2 minutes** (CPU-based async network calls, 0 GPU compute).
2. **Step 2 (Character Voice Conversion):** RVC (Retrieval-based Voice Conversion) with **RMVPE pitch extraction** transforms the speech into the target character's voice (e.g. Naruto) using pre-trained `.pth` and `.index` files in **~8–12 minutes** on T4 GPU.
3. **Step 3 (Lip-Sync & Muxing):** Phase-vocoder time-stretching aligns dialogue to exact SRT timestamps, composites the master track with 15-minute RAM safety flushes, and instantly muxes with the original video via `ffmpeg`.

---

## ⚡ Colab Quick Start

In a fresh Google Colab notebook cell:

```bash
# 1. Reset directory and clone NarutoGen
%cd /content
!rm -rf narutogen -narutogen
!git clone https://github.com/adarshraj990/-narutogen.git narutogen
%cd /content/narutogen

# 2. Install dependencies & FFmpeg
!pip install -q gradio edge-tts rvc-python pysrt pydub librosa soundfile psutil
!apt-get install -y ffmpeg

# 3. Launch Web App (with public gradio.live link)
!python dubbing_pipeline.py
```

---

## 🎨 Web Interface Features (`gr.Blocks`)

- **Subtitles Input:** Upload translated `.srt` files.
- **Video Input (Optional):** Upload raw anime video (`.mp4`, `.mkv`) for instant automatic video muxing.
- **RVC Character Voice:** Upload your character's `.pth` weights and optional `.index` feature file.
- **Pitch Transpose:** Fine-tune character pitch with the `-12` to `+12` semitone slider.
- **Dual Output:** Real-time audio player (`.wav`) + full dubbed video player (`.mp4`).
- **Public Share Link:** `demo.launch(share=True)` provides a remote web URL for Colab.
