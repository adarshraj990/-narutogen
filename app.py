"""
====================================================================================================
🍥 PRODUCTION MULTI-LANGUAGE DUBBING PIPELINE (HUGGING FACE SPACES ENTRYPOINT)
====================================================================================================
Kokoro-82M TTS + RVC Voice Conversion with 2-Hour OOM-Safe Chunking & Sequential Multi-Language Loop.
Optimized for Hugging Face Spaces (CPU/Zero-GPU/T4 GPU) and Local Execution.
====================================================================================================
"""

import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Backward compatibility shim for Gradio 4.x with modern huggingface_hub
try:
    import huggingface_hub
    if not hasattr(huggingface_hub, "HfFolder"):
        class _HfFolderShim:
            path_token = os.path.expanduser("~/.cache/huggingface/token")

            @classmethod
            def save_token(cls, token: str):
                try:
                    os.makedirs(os.path.dirname(cls.path_token), exist_ok=True)
                    with open(cls.path_token, "w", encoding="utf-8") as f:
                        f.write(token)
                except Exception:
                    pass

            @classmethod
            def get_token(cls):
                token = os.environ.get("HF_TOKEN")
                if token:
                    return token
                if os.path.exists(cls.path_token):
                    try:
                        with open(cls.path_token, "r", encoding="utf-8") as f:
                            return f.read().strip()
                    except Exception:
                        return None
                return None

            @classmethod
            def delete_token(cls):
                if os.path.exists(cls.path_token):
                    try:
                        os.remove(cls.path_token)
                    except OSError:
                        pass

        huggingface_hub.HfFolder = _HfFolderShim
except Exception:
    pass

# Backward compatibility patch for Gradio 4.44.1 with Starlette 1.0+ TemplateResponse (prevents 500 error)
try:
    from starlette.templating import Jinja2Templates
    _orig_template_response = Jinja2Templates.TemplateResponse

    def _safe_template_response(self, *args, **kwargs):
        # Gradio 4.44.1 calls: TemplateResponse(name, {"request": request, ...})
        # Starlette 1.0+ expects: TemplateResponse(request, name, context=...)
        if len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], dict):
            name = args[0]
            context = args[1]
            request = context.get("request")
            if request is not None:
                try:
                    return _orig_template_response(self, request, name, context, *args[2:], **kwargs)
                except TypeError:
                    pass
        return _orig_template_response(self, *args, **kwargs)

    Jinja2Templates.TemplateResponse = _safe_template_response
except Exception:
    pass

# Backward compatibility patch for Gradio 4.44.1 with modern Pydantic boolean schemas
try:
    import gradio_client.utils as _gc_utils
    _orig_js2py = _gc_utils._json_schema_to_python_type

    def _patched_js2py(schema, defs=None):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_js2py(schema, defs)

    _gc_utils._json_schema_to_python_type = _patched_js2py

    _orig_get_type = _gc_utils.get_type

    def _patched_get_type(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_get_type(schema)

    _gc_utils.get_type = _patched_get_type
except Exception:
    pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dubbing_pipeline import build_ui, CONFIGURED_RVC_MODEL_URL, launch_gradio_app, ensure_rvc_dependencies

# Synchronous dependency verification (removes detached daemon thread for Colab compatibility)
is_colab_env = "google.colab" in sys.modules or os.path.exists("/content")
try:
    ensure_rvc_dependencies()
except Exception as e:
    print(f"ℹ️ [STARTUP] RVC dependency verification notice: {e}")

# Gradio Interface
demo = build_ui()

if __name__ == "__main__":
    print("\n" + "=" * 80)
    if is_colab_env:
        print("🚀 [GOOGLE COLAB T4 GPU] Launching AI Dubbing Studio...")
    else:
        print("🚀 [AI DUBBING STUDIO] Launching AI Dubbing Studio...")
    print(f"🎙️ Configured Backend RVC Model: {CONFIGURED_RVC_MODEL_URL}")
    print("=" * 80 + "\n")
    launch_gradio_app(demo)

