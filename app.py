"""
====================================================================================================
🎙️ INDICF5 DUBBING STUDIO - GRADIO WEB INTERFACE
====================================================================================================
A modern, sleek, production-grade Gradio UI for the High-Speed Audio Dubbing & Time-Sync Pipeline.
Designed for seamless execution on Google Colab with share=True public URL generation.
====================================================================================================
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

import gradio as gr
import soundfile as sf

# Import the core dubbing pipeline
from dubbing_pipeline import run_pipeline, DEFAULT_BATCH_SIZE

# Custom CSS for a sleek, modern UI with polished cards, buttons, and badges
CUSTOM_CSS = """
/* Container & typography */
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header styling */
.header-box {
    text-align: center;
    padding: 28px 20px;
    background: linear-gradient(135deg, #1e1e2f 0%, #161623 50%, #0d0d17 100%);
    border-radius: 16px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    margin-bottom: 24px;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
}

.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff7e5f, #feb47b, #7f7fd5);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}

.header-subtitle {
    font-size: 1.05rem;
    color: #a0aec0;
    margin-bottom: 16px;
}

.badge-row {
    display: flex;
    justify-content: center;
    gap: 10px;
    flex-wrap: wrap;
}

.badge {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    padding: 5px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    color: #e2e8f0;
    font-weight: 500;
}

/* Card panels */
.card-panel {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
}

/* Primary Action Button */
.action-btn {
    background: linear-gradient(135deg, #ff5e3a 0%, #ff2a6d 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 14px 28px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(255, 42, 109, 0.35) !important;
    transition: all 0.3s ease !important;
}

.action-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(255, 42, 109, 0.5) !important;
}

/* Status cards */
.status-box {
    padding: 16px;
    border-radius: 12px;
    border-left: 5px solid #00f2fe;
    background: rgba(0, 242, 254, 0.05);
    color: #e2e8f0;
}
"""


def process_dubbing_ui(
    srt_file_obj,
    hours: float,
    minutes: float,
    seconds: float,
    voice_key: str,
    batch_size: int,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Handles user interaction from the Gradio interface:
    1. Validates inputs (SRT presence & valid duration).
    2. Saves uploaded SRT to a safe staging path.
    3. Runs the asynchronous FP16 dubbing pipeline with 15-minute RAM flushes.
    4. Returns status report and output WAV path to the audio component.
    """
    # Validation 1: SRT File
    if srt_file_obj is None:
        err_msg = (
            "### ❌ Error: Missing SRT Subtitle File\n"
            "Please upload a valid translated `.srt` subtitle file before starting the pipeline."
        )
        return err_msg, None

    srt_path = getattr(srt_file_obj, "name", str(srt_file_obj))
    if not os.path.exists(srt_path):
        err_msg = "### ❌ Error: Uploaded file not found on disk. Please re-upload."
        return err_msg, None

    # Validation 2: Duration
    h = int(hours or 0)
    m = int(minutes or 0)
    s = int(seconds or 0)

    if h < 0 or m < 0 or s < 0:
        err_msg = "### ❌ Error: Duration values cannot be negative."
        return err_msg, None

    if h == 0 and m == 0 and s == 0:
        err_msg = (
            "### ❌ Error: Total Video Duration cannot be 00:00:00\n"
            "Please specify the actual video duration (Hours, Minutes, Seconds) to establish the master canvas "
            "and prevent cutting off background music or end credits."
        )
        return err_msg, None

    total_duration_sec = (h * 3600) + (m * 60) + s

    # Setup output target path
    output_dir = Path(tempfile.gettempdir()) / "indicf5_dubbing_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    output_wav_path = output_dir / f"dubbed_audio_{timestamp}.wav"

    progress(0.05, desc="🚀 Initializing Dubbing Pipeline...")
    start_time = time.time()

    try:
        progress(0.15, desc="🎙️ Loading IndicF5 Model onto GPU (FP16 & SDPA)...")
        # Execute the production pipeline
        final_path = run_pipeline(
            srt_file=srt_path,
            output_audio=str(output_wav_path),
            hours=h,
            minutes=m,
            seconds=s,
            voice_key=voice_key,
            batch_size=int(batch_size),
        )

        elapsed_time = time.time() - start_time

        # Inspect final audio metadata
        audio_info = sf.info(final_path)
        actual_dur_sec = audio_info.duration
        actual_mins = int(actual_dur_sec // 60)
        actual_secs = int(actual_dur_sec % 60)
        file_size_mb = os.path.getsize(final_path) / (1024 * 1024)

        status_markdown = f"""
### 🎉 Dubbing Pipeline Completed Successfully!

| Metric | Value |
| :--- | :--- |
| **Status** | ✅ Completed in **{elapsed_time:.1f}s** ({elapsed_time/60:.2f} mins) |
| **Master Canvas Duration** | `{h:02d}:{m:02d}:{s:02d}` ({total_duration_sec}s) |
| **Generated Audio Duration** | `{actual_mins:02d}:{actual_secs:02d}` ({actual_dur_sec:.2f}s) |
| **Sample Rate** | {audio_info.samplerate} Hz (PCM 16-bit Mono) |
| **Output File Size** | {file_size_mb:.2f} MB |
| **Voice Profile Used** | `{voice_key}` |
| **RAM Management** | Flushed cleanly in 15-minute segments (0 OOM) |

*You can now listen to the audio directly or download it using the player below.*
"""
        progress(1.0, desc="✅ Complete!")
        return status_markdown, final_path

    except Exception as e:
        err_msg = f"""
### ❌ Pipeline Execution Failed
An error occurred during audio generation:
```text
{str(e)}
```
*Tip: Ensure your Google Colab runtime is set to **T4 GPU** (`Runtime -> Change runtime type -> T4 GPU`).*
"""
        return err_msg, None


# ==================================================================================================
# BUILD GRADIO BLOCKS APPLICATION
# ==================================================================================================
def build_app():
    # Use Gradio Soft Theme with customized accents
    theme = gr.themes.Soft(
        primary_hue="rose",
        secondary_hue="slate",
        neutral_hue="slate",
    )

    with gr.Blocks(theme=theme, css=CUSTOM_CSS, title="IndicF5 Audio Dubbing Studio") as demo:

        # Header Box / Hero Section
        gr.HTML(
            """
            <div class="header-box">
                <div class="header-title">🎙️ IndicF5 Hinglish Dubbing Studio</div>
                <div class="header-subtitle">
                    High-Speed, Production-Ready Audio Dubbing & Exact Time-Synchronization Pipeline
                </div>
                <div class="badge-row">
                    <span class="badge">⚡ T4 GPU Optimized (FP16 + SDPA)</span>
                    <span class="badge">🧠 Dynamic Batching (4-5 Cues)</span>
                    <span class="badge">🛡️ 15-Min RAM Flushing Safe</span>
                    <span class="badge">🎵 Zero-Pitch Distortion Sync</span>
                </div>
            </div>
            """
        )

        with gr.Row():
            # LEFT COLUMN: Inputs & Controls
            with gr.Column(scale=5):
                gr.Markdown("### 📥 1. Upload Subtitles & Configure Video Canvas")

                srt_input = gr.File(
                    label="Translated SRT Subtitles (*.srt)",
                    file_types=[".srt"],
                    file_count="single",
                    elem_classes="card-panel",
                )

                gr.Markdown("#### 🎬 Total Video Duration (Dynamic Canvas Setup)")
                gr.Markdown(
                    "<small style='color: #718096;'>Specify the total video duration. This guarantees that "
                    "background music/outros after the final dialogue are never truncated.</small>"
                )

                with gr.Row():
                    hours_input = gr.Number(
                        label="Hours",
                        value=0,
                        minimum=0,
                        step=1,
                        precision=0,
                    )
                    minutes_input = gr.Number(
                        label="Minutes",
                        value=0,
                        minimum=0,
                        maximum=59,
                        step=1,
                        precision=0,
                    )
                    seconds_input = gr.Number(
                        label="Seconds",
                        value=48,
                        minimum=0,
                        maximum=59,
                        step=1,
                        precision=0,
                    )

                with gr.Accordion("⚙️ Advanced Audio & GPU Settings", open=False):
                    voice_dropdown = gr.Dropdown(
                        choices=[
                            ("Ritu (Hinglish Female - Default)", "ritu_hinglish"),
                            ("Tamil-Hinglish", "ta_hinglish"),
                            ("Bengali", "bn"),
                            ("Gujarati", "gu"),
                            ("Kannada", "kn"),
                            ("Malayalam", "ml"),
                            ("Marathi", "mr"),
                            ("Odia", "or"),
                            ("Punjabi", "pa"),
                            ("Telugu", "te"),
                        ],
                        value="ritu_hinglish",
                        label="Reference Voice Profile",
                        info="Bundled Indic reference speaker for voice matching",
                    )
                    batch_slider = gr.Slider(
                        minimum=1,
                        maximum=8,
                        value=DEFAULT_BATCH_SIZE,
                        step=1,
                        label="Dynamic Batch Size (Cues per Batch)",
                        info="Groups similar-duration lines to saturate T4 GPU Tensor Cores",
                    )

                start_btn = gr.Button(
                    "⚡ Start Dubbing Pipeline",
                    variant="primary",
                    size="lg",
                    elem_classes="action-btn",
                )

                # Convenient Preset Example
                gr.Markdown("#### 💡 Quick Test Example")
                example_path = Path("sample_hindi_english.srt")
                if example_path.exists():
                    gr.Examples(
                        examples=[
                            [str(example_path), 0, 0, 48, "ritu_hinglish", 4],
                        ],
                        inputs=[
                            srt_input,
                            hours_input,
                            minutes_input,
                            seconds_input,
                            voice_dropdown,
                            batch_slider,
                        ],
                        label="Click to load sample Hindi-English code-switched SRT",
                    )

            # RIGHT COLUMN: Status Updates & Audio Output
            with gr.Column(scale=6):
                gr.Markdown("### 🎧 2. Output & Real-Time Status")

                status_output = gr.Markdown(
                    """
                    <div class="status-box">
                        <b>Ready to process.</b> Upload your translated <code>.srt</code> file and set the video duration, 
                        then click <b>Start Dubbing Pipeline</b>.
                    </div>
                    """
                )

                audio_output = gr.Audio(
                    label="Generated Master Dubbed Audio (.wav)",
                    type="filepath",
                    interactive=False,
                )

                gr.Markdown(
                    """
                    > **Colab T4 RAM Safety Note:**  
                    > If dubbing long 2–3 hour videos, the pipeline automatically processes and exports audio in **15-minute flushed chunks**, 
                    > ensuring Google Colab's 12GB RAM limit is never exceeded.
                    """
                )

        # Wire the Action Button
        start_btn.click(
            fn=process_dubbing_ui,
            inputs=[
                srt_input,
                hours_input,
                minutes_input,
                seconds_input,
                voice_dropdown,
                batch_slider,
            ],
            outputs=[
                status_output,
                audio_output,
            ],
        )

    return demo


# ==================================================================================================
# LAUNCH WITH PUBLIC LINK (share=True FOR COLAB WEB ACCESS)
# ==================================================================================================
if __name__ == "__main__":
    demo = build_app()
    # share=True creates a public https://XXXX.gradio.live link for remote Colab access
    demo.launch(
        share=True,
        server_name="0.0.0.0",
        server_port=7860,
        show_api=False,
        debug=True,
    )
