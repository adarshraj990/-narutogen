---
title: Audiogenflow
emoji: 🎙️
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 4.44.1
python_version: 3.11
app_file: app.py
pinned: false
---

# 🎙️ AudioGenFlow: Automated Multi-Language AI Video Dubbing Studio
### High-Fidelity Neural Dubbing (Kokoro-82M TTS + RVC RMVPE + 2-Hour Chunking Architecture)
**100% Free & Open Source | Zero Paid APIs | Zero OOM Crashes | Native Hugging Face Spaces**

---

## 🌟 Key Architecture & Capabilities

1. **100% Local Kokoro-82M Neural TTS:**
   - Replaced all external cloud TTS with **Kokoro-82M** running completely offline.
   - High-fidelity natural prosody for multi-language synthesis (**Hindi, Spanish, French, Portuguese**).

2. **2-Hour Chunking Architecture (Safe for HF Spaces):**
   - Automatically breaks down long SRT files/scripts into **5-minute sequential slices** (`CHUNK_DURATION_MINUTES = 5`).
   - Processes each chunk through Kokoro TTS & RVC, saves audio to disk, and forcefully clears memory (`gc.collect()`, `torch.cuda.empty_cache()`).
   - Total RAM usage is capped under **1.5 GB**, allowing 2-hour long files to process safely without triggering Hugging Face container restarts.

3. **Lossless FFmpeg Stream Assembly:**
   - Chunks are stitched together instantaneously via FFmpeg stream copy (`-c copy`) without CPU/GPU re-encoding.
   - Includes an in-memory-safe block-by-block `soundfile` streaming fallback.

4. **Sequential Multi-Language Processing:**
   - Select multiple target languages in a single run.
   - The backend processes them sequentially: finishes Language 1 (all chunks + master stitch), then automatically starts Language 2.

5. **Retrieval-based Voice Conversion (RVC V2):**
   - Converts base Kokoro audio to character voices using **RMVPE pitch extraction**.
   - Backend URL placeholder to configure your character voice model (`.zip` containing `.pth` and `.index`).

---

## 🎛️ Minimalist Gradio Web UI

The interface contains strictly the essential controls:
- **1. Subtitle Upload:** Single `.srt` file.
- **2. Video Duration:** Numeric inputs for `Hours`, `Minutes`, and `Seconds` (sets timeline boundaries and prevents cutting off background music or credits).
- **3. Target Languages:** Multi-choice checkboxes (`Hindi`, `Spanish`, `French`, `Portuguese`).
- **4. Action Button:** Prominent **"Start Dubbing"** button.

---

## 🌐 Live Hugging Face Space

The pipeline is live and accessible at:  
👉 **[https://huggingface.co/spaces/adarshraj990/audiogenflow](https://huggingface.co/spaces/adarshraj990/audiogenflow)**

Whenever changes are pushed to `main`, Hugging Face automatically rebuilds and deploys the container.

---

## ⚙️ Changing Character Voice Model (Backend URL)

To change the RVC character model, edit `CONFIGURED_RVC_MODEL_URL` in [dubbing_pipeline.py](file:///c:/Users/Adarsh/Desktop/NARUTOGEN/dubbing_pipeline.py#L132):

```python
# 📌 [INSERT RVC MODEL DOWNLOAD LINK HERE]:
CONFIGURED_RVC_MODEL_URL = "https://huggingface.co/path-to-your-rvc-model.zip"
```
The pipeline automatically downloads, verifies, and unpacks the `.pth` and `.index` files on launch.
