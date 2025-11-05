"""
Configuration settings for the Chunkformer Streamlit app
"""

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "Chunkformer Video Transcription",
    "page_icon": "🎬",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Model options
MODEL_OPTIONS = [
    "khanhld/chunkformer-ctc-large-vie",
    "khanhld/chunkformer-rnnt-large-vie",
    "khanhld/chunkformer-large-en-libri-960h",
]

# Supported video formats
SUPPORTED_VIDEO_FORMATS = ["mp4", "avi", "mov", "mkv", "webm"]

# Processing parameters
CHUNK_SIZE = 64
LEFT_CONTEXT_SIZE = 128
RIGHT_CONTEXT_SIZE = 128
TOTAL_BATCH_DURATION = 1800
MAX_SILENCE_DURATION = 0.5  # Default max silence duration in seconds

# Timeouts
FFMPEG_TIMEOUT = 600  # 10 minutes
UPLOAD_CHUNK_SIZE = 4 * 1024 * 1024  # 4 MB

# UI parameters
VIDEO_PLAYER_HEIGHT = 620
TRANSCRIPT_CONTEXT_WINDOW = 5.0

# Session state keys
SESSION_STATE_RESULTS = "chunkformer_results"
SESSION_STATE_UPLOAD_KEY = "chunkformer_upload_key"

# Environment variables
ENV_DISABLE_TQDM = "TQDM_DISABLE"
