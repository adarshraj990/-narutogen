# 🍥 AI Dubbing Studio — Colab Deployment Guide

> **Kokoro-82M TTS + RVC Voice Conversion (RMVPE) + FFmpeg + Gradio**
> Optimized for Google Colab Free T4 GPU

---

## 🚀 One-Click Launch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/adarshraj990/-narutogen/blob/main/AI_Dubbing_Studio_Colab.ipynb)

---

## ✅ Dependency Matrix (Conflict-Free)

| Package | Version | Role | Notes |
|---|---|---|---|
| `numpy` | `<2.0.0` | ABI compat guard | Must be first |
| `infer-rvc-python` | latest | RVC engine | **fairseq-FREE** — no compile hang |
| `pyworld-prebuilt` | latest | Pitch tools | Pre-built binary — no Cython compile |
| `onnxruntime-gpu` | latest | RMVPE on CUDA | T4 GPU accelerated |
| `faiss-gpu-cu12` | latest | Index search | CUDA 12 T4 wheel |
| `kokoro` | latest | Kokoro-82M TTS | Local, zero API keys |
| `hydra-core` + `omegaconf` | latest | RVC config | Pure Python |
| `antlr4-python3-runtime` | `==4.9.3` | Grammar parser | Version-pinned |

> **Why `infer-rvc-python` instead of `rvc-python`?**
> `rvc-python` depends on `fairseq` which requires C++ Cython compilation → **hangs forever** in Colab.
> `infer-rvc-python` is a maintained fork with fairseq removed → installs in < 60 seconds.

---

## 📋 Step-by-Step Instructions

### Step 0 — Set GPU Runtime
`Runtime → Change runtime type → Hardware accelerator → T4 GPU → Save`

### Step 1 — Verify GPU
Run **Cell 1**: confirms `nvidia-smi` output and asserts `torch.cuda.is_available()`.

### Step 2 — Install Dependencies (~ 3-5 min)
Run **Cell 2**: installs all packages. Watch for the final line:
```
✅ infer_rvc_python.RVCInference: <class '...'>
```
If you see `🚨 infer_rvc_python failed to import!` — read the traceback and fix it before continuing.

### Step 3 — Launch Studio
Run **Cell 3**: starts `python app.py`.
- ✅ Generates a public `https://xxxx.gradio.live` link
- ✅ Streams all pipeline logs to cell output
- ✅ Saves output audio to `/content/narutogen/outputs/`

> **If you see `infer_rvc_python` ImportError in Cell 3:**
> `Runtime → Restart runtime` → Skip Cell 2 → Run Cell 3 directly.
> (pip installs persist across restarts in the same Colab session)

---

## 🔑 Configure Your RVC Character Model

Edit `dubbing_pipeline.py`, line ~362:
```python
CONFIGURED_RVC_MODEL_URL = (
    "https://huggingface.co/YOUR_USERNAME/YOUR_MODEL_REPO/resolve/main/your_model.zip"
)
```
The ZIP must contain a `.pth` (weights) and optionally a `.index` (feature index) file.

---

## 🧠 Architecture Notes

```
SRT File
   └─→ [Kokoro-82M TTS]     → per-cue WAV (24kHz, mono)
          └─→ [RVC RMVPE]   → character voice WAV (16kHz in, 44.1kHz out)
                 └─→ [FFmpeg] → time-synced master WAV
```

- **5-min chunking** prevents OOM on 2-hour videos
- **Sequential multi-language** processing (Hindi → Spanish → French → Portuguese)
- **`torch.cuda.empty_cache()` + `gc.collect()`** after every chunk
- **No silent fallbacks** — every error raises loudly with full traceback
