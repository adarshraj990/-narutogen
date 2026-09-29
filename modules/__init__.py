"""
NarutoGen Audio Pipeline Modules
"""
from .tts_generator import (
    generate_base_speech,
    parse_srt_file,
    SubtitleCue,
    SUPPORTED_VOICES,
    resolve_voice_name,
)
from .rvc_converter import (
    RVCBatchConverter,
    run_rvc_conversion,
    convert_base_audio_to_naruto,
)
from .audio_stitcher import AudioStitcher, stitch_audio_chunks
from .model_downloader import (
    ensure_naruto_model,
    DEFAULT_NARUTO_MODEL_URL,
    download_file_with_progress,
)

__all__ = [
    "generate_base_speech",
    "parse_srt_file",
    "SubtitleCue",
    "SUPPORTED_VOICES",
    "resolve_voice_name",
    "RVCBatchConverter",
    "run_rvc_conversion",
    "convert_base_audio_to_naruto",
    "AudioStitcher",
    "stitch_audio_chunks",
    "ensure_naruto_model",
    "DEFAULT_NARUTO_MODEL_URL",
    "download_file_with_progress",
]
