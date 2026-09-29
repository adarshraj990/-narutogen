# 🍥 NarutoGen - Google Colab Setup Guide

Follow these exact steps in Google Colab (Free Tier T4 GPU).

---

### Step 0: Set Runtime to GPU
1. In Google Colab top menu, click **Runtime** (रनटाइम) &rarr; **Change runtime type** (रनटाइम प्रकार बदलें).
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

### Cell 2: Install Dependencies & RVC GPU Support
```bash
!apt-get update -qq && apt-get install -y ffmpeg
!pip install -q -r requirements.txt
!pip install -q rvc-python
```

---

### Cell 3: Download & Setup Naruto RVC Model
Automatically downloads `naruto-uzumaki.zip` (56 MB) from Hugging Face and extracts it into `models/naruto/`, `weights/`, and `logs/naruto/`:
```bash
!python -m modules.model_downloader
```

---

### Cell 4: Run (Choose Option A or Option B)

#### Option A: Launch Interactive Web UI (Recommended)
```bash
!python dubbing_pipeline.py
```
*Click the public `https://xxxx.gradio.live` link generated in the terminal to upload SRTs, adjust pitch, select languages (Hindi, Spanish, French, Portuguese), and download the master audio.*

#### Option B: Run Fast Command-Line Pipeline
```bash
# Dub Hindi SRT:
!python pipeline.py --input inputs/sample_hindi_english.srt --output outputs/naruto_master.wav

# Dub with Portuguese base voice:
!python pipeline.py --input inputs/sample_hindi_english.srt --voice pt_antonio --output outputs/naruto_master_pt.wav
```
