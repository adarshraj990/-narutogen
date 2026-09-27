# 🎙️ IndicF5 Hinglish Audio Dubbing & Time-Sync Pipeline
### Optimized for Google Colab Free Tier (T4 GPU, 15GB VRAM, 12GB RAM, 2 vCPUs)

---

## ⚡ Quick Start on Google Colab

### Step 1: Set Runtime to T4 GPU
1. In Google Colab, click **Runtime** in the top menu.
2. Select **Change runtime type**.
3. Choose **T4 GPU** and click **Save**.

### Step 2: Install Dependencies (Run in first Colab cell)
```bash
!pip install -q gradio transformers torchdiffeq x-transformers vocos soundfile librosa pysrt pydub psutil
!apt-get install -y ffmpeg
```

### Step 3: Launch the Gradio Web Interface (Recommended)
```bash
!python app.py
```
*Gradio will output a public URL (`https://XXXXX.gradio.live`) so you can access the sleek UI directly from your browser!*

### Or Run via Command-Line Interface (CLI):
```bash
!python dubbing_pipeline.py --srt sample_hindi_english.srt
```
*You will be interactively prompted for the total video duration (Hours, Minutes, Seconds).*

---

## 🏗️ Architecture Overview

```
                      +-----------------------------+
                      |   Translated SRT Subtitles  |
                      +--------------+--------------+
                                     |
                [Localized 8-12 Subtitle Duration Sorting]
                                     |
                          +----------v----------+
                          |   Dynamic Batches   |
                          |  (4-5 Similar Cues) |
                          +----------+----------+
                                     |
                                     | (Concurrent Threads)
      +------------------------------+------------------------------+
      |                                                             |
+-----v----------------------+                        +-------------v---------------+
|     PRODUCER THREAD (GPU)  |                        |    CONSUMER THREAD (CPU)    |
| - Tharshan/indicf5_...     |                        | - Pulls from Queue          |
| - FP16 Precision (Half)    |                        | - Pitch-Preserving Speedup  |
| - SDPA Attention Fast-Path |                        |   or Silence Padding        |
| - Continuous Generation    |                        | - Overlays onto 15-min      |
+-------------+--------------+                        |   Active Window Canvas      |
              |                                       +-------------+---------------+
              |  Pushes audio arrays                                |
              +------------------> [Thread-Safe Queue] <------------+
                                   (Maxsize: 12)
                                                                    |
                                                      +-------------v---------------+
                                                      |  RAM Flushing (Every 15min) |
                                                      | - Exports part_001.wav      |
                                                      | - del canvas; gc.collect()  |
                                                      | - Prevents 12GB Colab OOM   |
                                                      +-------------+---------------+
                                                                    |
                                                      +-------------v---------------+
                                                      |  Final Concat (0 MB RAM)    |
                                                      | - ffmpeg stream copy        |
                                                      | - Matches exact user length |
                                                      +-----------------------------+
```

---

## 📊 Key Highlights & Optimizations

1. **FP16 & SDPA on T4 GPU**:
   - The DiT diffusion transformer runs in half precision (`torch.float16`).
   - PyTorch's native `scaled_dot_product_attention` enables FlashAttention / memory-efficient kernels, maximizing generation speed.
   - VRAM usage is strictly confined to ~1.2 GB, well below Colab's 15 GB ceiling.

2. **Asynchronous Producer-Consumer**:
   - The GPU never idles waiting for CPU audio time-stretching or disk I/O.
   - The CPU never blocks the GPU from continuing to next batches.
   - A bounded queue (`maxsize=12`) prevents backpressure RAM accumulation.

3. **Strict 15-Minute RAM Flush (No OOM on 3-Hour Audio)**:
   - For a 3-hour video, standard Pydub canvas overlay creates hundreds of large in-memory byte buffers, causing Colab to hit the 12GB RAM limit and crash (SIGKILL).
   - Our pipeline isolates each 15-minute chunk, overlays dialogues, flushes the slice to disk as `part_001.wav`, and explicitly purges the memory variable (`gc.collect()`).

4. **Dynamic Canvas & Outro Protection**:
   - The user inputs the total video duration (Hours, Minutes, Seconds).
   - If the last dialogue finishes before the video ends, the pipeline automatically pads the final window with silence to match the exact video length, ensuring background music and end credits are preserved.

5. **Pitch-Preserving Time Synchronization**:
   - If generated audio exceeds the subtitle duration, librosa's phase-vocoder time-stretch speeds up the dialogue without pitch shift (no chipmunk effect).
   - If generated audio is shorter, it is padded with silence to achieve sample-accurate alignment.
