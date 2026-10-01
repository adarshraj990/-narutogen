# 🍥 NarutoGen - Google Colab & Hugging Face Spaces Setup Guide

Follow these exact steps in Google Colab (Free Tier T4 GPU) or Hugging Face Spaces to run the Kokoro-82M + RVC Voice Dubbing Pipeline with **2-Hour Chunking Architecture**.

---

### Step 0: Set Runtime to GPU (Colab)
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

### Cell 2: Install Dependencies & RVC GPU Support (Conflict-Free Matrix)
```bash
!apt-get update -qq && apt-get install -y ffmpeg espeak-ng

# 1. Enforce NumPy < 2.0.0 ABI compatibility for Fairseq / PyTorch / PyWorld C-extensions
!pip install -q "numpy<2.0.0"

# 2. Install prebuilt C-extensions for Fairseq & PyWorld without triggering PyPI source build failure
!pip install -q --no-deps fairseq-fixed pyworld-fixed rvc-python

# 3. Install GPU-accelerated FAISS (CUDA 12) & ONNX Runtime GPU (T4 TensorRT/CUDA)
!pip install -q faiss-gpu-cu12 onnxruntime-gpu || pip install -q faiss-cpu onnxruntime-gpu

# 4. Install Fairseq/RVC sub-dependencies, Kokoro Neural TTS & Web UI
!pip install -q hydra-core omegaconf "antlr4-python3-runtime==4.9.3" kokoro soundfile pydub pysrt gradio>=4.44.1 librosa huggingface_hub requests psutil
```

---

### Cell 3: Download & Setup Configured RVC Model (Optional Standalone)
Automatically downloads the configured RVC model (CarryMinati / Naruto / custom voice) from Hugging Face into `models/character/` and `weights/` (the Web UI will also auto-download if missing):
```bash
!python -m modules.model_downloader
```

---

### Cell 4: Launch Minimalist Gradio Web UI
```bash
!python dubbing_pipeline.py
```
*Click the public `https://xxxx.gradio.live` link generated in the terminal to upload your `.srt` file, set video duration, select your desired languages (Hindi, Spanish, French, Portuguese), and click **Start Dubbing**!*

---

### 🌐 Deploying to Hugging Face Spaces
1. Create a new Space on [Hugging Face](https://huggingface.co/new-space) (SDK: **Gradio**, Hardware: **Free CPU / Zero-GPU / T4 Small**).
2. Connect your GitHub repository `https://github.com/adarshraj990/-narutogen` or push the files directly.
3. Hugging Face Spaces will automatically launch `app.py`. The built-in 5-minute chunking engine guarantees safe execution without running into 16GB RAM Out-of-Memory (OOM) errors even on 2-hour long files!
