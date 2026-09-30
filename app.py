"""
====================================================================================================
🍥 PRODUCTION MULTI-LANGUAGE DUBBING PIPELINE (HUGGING FACE SPACES ENTRYPOINT)
====================================================================================================
Kokoro-82M TTS + RVC Voice Conversion with 2-Hour OOM-Safe Chunking & Sequential Multi-Language Loop.
Optimized for Hugging Face Spaces (CPU/Zero-GPU/T4 GPU) and Google Colab.
====================================================================================================
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dubbing_pipeline import build_ui, CONFIGURED_RVC_MODEL_URL

# Hugging Face Spaces detects demo at module level
demo = build_ui()

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("🚀 [HUGGING FACE SPACES / WEB APP] Launching AI Dubbing Interface...")
    print(f"🎙️ Configured Backend RVC Model: {CONFIGURED_RVC_MODEL_URL}")
    print("=" * 80 + "\n")

    # In Hugging Face Spaces (SPACE_ID is automatically populated), do not enable share=True
    is_hf_space = os.getenv("SPACE_ID") is not None
    demo.queue().launch(
        share=not is_hf_space,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
