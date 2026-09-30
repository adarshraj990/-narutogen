# 🍥 NarutoGen - Google Colab Setup Guide

Follow these exact steps in Google Colab (Free Tier T4 GPU) to run the Kokoro-82M + RVC Voice Dubbing Pipeline.

---

### Step 0: Set Runtime to GPU
1. In Google Colab top menu, click **Runtime** &rarr; **Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU**.
3. Click **Save**.

---

### Cell 1: Clone Repository & Update
```python
import os
if not os.path.exists('/content/narutogen'):
    !git clone https://github.com/adarshraj990/-narutogen.git /content/narutogen
%cd /content/narutogen
!git pull origin main
```

---

### Cell 2: Install Dependencies & RVC GPU Support (Conflict-Free)
```bash
!apt-get update -qq && apt-get install -y ffmpeg espeak-ng
!pip install -q kokoro soundfile misaki gradio pydub pysrt librosa
!pip install -q av ffmpeg-python loguru praat-parselmouth torchcrepe faiss-cpu
!pip install -q --no-deps fairseq-fixed pyworld-fixed rvc-python
```

---

### Cell 3: Download & Setup Configured RVC Model
Automatically downloads the configured RVC model (CarryMinati / Naruto / custom voice) from Hugging Face into `models/character/` and `weights/`:
```bash
!python -m modules.model_downloader
```

---

### Cell 4: Launch Minimalist Gradio Web UI
```bash
!python dubbing_pipeline.py
```
*Click the public `https://xxxx.gradio.live` link generated in the terminal to upload your `.srt` file, set video duration, pick your target language (Hindi, Spanish, French, Portuguese), and click **Start Dubbing**!*

