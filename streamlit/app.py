"""
Streamlit App for Chunkformer-based Video Transcription with Synchronized Playback

This app allows users to:
1. Upload video files
2. Run transcription using Chunkformer model
3. Display video with synchronized transcription and timestamps
"""

import json
import logging
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from audio_processing import save_uploaded_file_with_progress
from config import (
    CHUNK_SIZE,
    ENV_DISABLE_TQDM,
    LEFT_CONTEXT_SIZE,
    MAX_SILENCE_DURATION,
    MODEL_OPTIONS,
    PAGE_CONFIG,
    RIGHT_CONTEXT_SIZE,
    SESSION_STATE_RESULTS,
    SESSION_STATE_UPLOAD_KEY,
    SUPPORTED_VIDEO_FORMATS,
    TOTAL_BATCH_DURATION,
    VIDEO_PLAYER_HEIGHT,
)
from transcription import load_model, transcribe_audio
from ui_components import (
    render_custom_css,
    render_footer,
    render_hero_section,
    render_landing_page,
    render_synchronized_player,
)
from utils import create_subtitle_srt, format_timestamp, guess_video_mime_type

import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Suppress tqdm output in Streamlit
os.environ[ENV_DISABLE_TQDM] = "1"

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def render_sidebar():
    """Render the sidebar with configuration options"""
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.markdown("---")

        st.subheader("🤖 Model Selection")
        model_choice = st.selectbox(
            "Choose your model:",
            MODEL_OPTIONS,
            help=("CTC: Faster and more efficient | " "RNNT: More accurate but slower"),
        )

        st.markdown("---")
        st.subheader("🎚️ Advanced Settings")

        max_silence_duration = st.slider(
            "Max Silence Duration (seconds)",
            min_value=0.1,
            max_value=2.0,
            value=MAX_SILENCE_DURATION,
            step=0.1,
            help="Maximum duration of silence to detect sentence breaks.  \
                    Lower values create more segments.",
        )

        st.markdown("---")
        st.subheader("ℹ️ About")
        st.markdown(
            """
        **Chunkformer** is a state-of-the-art Vietnamese ASR model
        built on streaming-compatible architecture.

        - Fast and accurate transcription
        - Streaming-ready architecture
        - Support for multiple video formats
        """
        )

        return model_choice, max_silence_duration


def render_results_section(results: dict, uploaded_file):
    """Render the results section with video player and download options

    Args:
        results: Dictionary containing transcription results
        uploaded_file: Streamlit uploaded file object
    """
    segments = results["segments"]
    full_transcript = results["full_transcript"]
    file_name = uploaded_file.name

    st.markdown("---")
    st.subheader("🎬 Video & Synchronized Transcript")
    st.markdown(
        "*Transcript highlights update as the video plays. Click any line to seek the video.*"
    )

    with st.spinner("⏳ Rendering video player and transcript..."):
        try:
            video_data = uploaded_file.getvalue()
        except Exception:
            video_data = None

        render_synchronized_player(
            video_bytes=video_data,
            mime_type=guess_video_mime_type(file_name),
            segments=segments,
            height=VIDEO_PLAYER_HEIGHT,
        )

    # Export options
    st.markdown("---")
    st.subheader("💾 Export Results")
    st.markdown("Download your transcription in multiple formats:")

    col1, col2, col3 = st.columns(3)

    with col1:
        json_data = json.dumps(segments, indent=2)
        st.download_button(
            label="📄 JSON (Full Data)",
            data=json_data,
            file_name="transcription.json",
            mime="application/json",
            use_container_width=True,
        )

    with col2:
        srt_content = create_subtitle_srt(segments)
        st.download_button(
            label="🎬 SRT (Subtitles)",
            data=srt_content,
            file_name="subtitles.srt",
            mime="text/plain",
            use_container_width=True,
        )

    with col3:
        st.download_button(
            label="📝 TXT (Transcript)",
            data=full_transcript,
            file_name="transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # Statistics
    st.markdown("---")
    st.subheader("📊 Transcription Statistics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Segments", len(segments), delta=None)

    with col2:
        total_duration = results["duration"]
        st.metric("Duration", format_timestamp(total_duration), delta=None)

    with col3:
        st.metric("Word Count", results["word_count"], delta=None)

    with col4:
        st.metric("Character Count", results["char_count"], delta=None)


def process_video(uploaded_file, model_choice: str, max_silence_duration: float):
    """Process the uploaded video file

    Args:
        uploaded_file: Streamlit uploaded file object
        model_choice: Selected model path
        max_silence_duration: Maximum silence duration for sentence break detection
    """
    st.session_state[SESSION_STATE_RESULTS] = None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            logger.info(f"[MAIN] Created temporary directory: {tmpdir}")

            # Create progress tracking
            progress_container = st.container()
            with progress_container:
                st.subheader("⏳ Processing Progress")

            file_name = uploaded_file.name
            video_extension = Path(file_name).suffix or ".mp4"
            video_path = os.path.join(tmpdir, f"video_file{video_extension}")
            logger.info(f"[MAIN] Video path: {video_path}")

            # Step 1: Save file
            with progress_container:
                progress_placeholder = st.empty()
                status_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0)

            with status_placeholder:
                st.markdown("**Step 1/4:** Saving file...")

            save_elapsed = save_uploaded_file_with_progress(
                uploaded_file=uploaded_file,
                destination_path=video_path,
                progress_bar=progress_bar,
                status_placeholder=status_placeholder,
            )
            with progress_placeholder:
                progress_bar.progress(25)
            with status_placeholder:
                st.markdown(f"✅ **Step 1/4:** File saved in {save_elapsed:.2f}s")

            # Step 2: Load model
            logger.info(f"[MAIN] Loading model: {model_choice}")
            import time

            model_start = time.time()

            with status_placeholder:
                st.markdown("**Step 2/4:** Loading model...")

            with st.spinner(f"Loading {model_choice.split('/')[-1]} model..."):
                model = load_model(model_choice)
                model_elapsed = time.time() - model_start
                if model is None:
                    logger.error("[MAIN] Failed to load model")
                    st.error("❌ Failed to load model")
                    st.stop()

            with progress_placeholder:
                progress_bar.progress(50)
            with status_placeholder:
                st.markdown(f"✅ **Step 2/4:** Model loaded in {model_elapsed:.2f}s")

            # Step 3: Transcribe
            logger.info("[MAIN] Starting transcription...")
            transcribe_start = time.time()

            with status_placeholder:
                st.markdown("**Step 3/4:** Transcribing (this may take a while)...")

            with st.spinner("🎤 Transcribing media..."):
                segments, full_transcript = transcribe_audio(
                    model=model,
                    media_path=video_path,
                    chunk_size=CHUNK_SIZE,
                    left_context_size=LEFT_CONTEXT_SIZE,
                    right_context_size=RIGHT_CONTEXT_SIZE,
                    total_batch_duration=TOTAL_BATCH_DURATION,
                    max_silence_duration=max_silence_duration,
                )

            transcribe_elapsed = time.time() - transcribe_start

            # Step 4: Finalize
            if not segments:
                logger.error("[MAIN] Transcription returned no segments")
                st.error("❌ Failed to get transcription results")
            else:
                logger.info(f"[MAIN] Transcription successful: {len(segments)} segments")

                with progress_placeholder:
                    progress_bar.progress(75)
                with status_placeholder:
                    st.markdown(
                        f"✅ **Step 3/4:** Transcribed {len(segments)} "
                        f"segments in {transcribe_elapsed:.2f}s"
                    )

                with status_placeholder:
                    st.markdown("**Step 4/4:** Finalizing...")

                st.session_state[SESSION_STATE_RESULTS] = {
                    "segments": segments,
                    "full_transcript": full_transcript,
                    "word_count": len(full_transcript.split()),
                    "char_count": len(full_transcript),
                    "duration": segments[-1]["end"] if segments else 0,
                    "model_choice": model_choice,
                    "timestamp": datetime.utcnow().isoformat(),
                }

                with progress_placeholder:
                    progress_bar.progress(100)
                with status_placeholder:
                    st.markdown("✅ **Step 4/4:** Complete! Ready to review.")

                progress_container.empty()
                st.success(
                    f"✅ Transcription complete! ({len(segments)} segments "
                    f"in {transcribe_elapsed:.2f}s)"
                )

    except Exception as exc:
        logger.error("[MAIN] Processing error", exc_info=True)
        st.error(f"❌ Processing error: {exc}")
        st.session_state[SESSION_STATE_RESULTS] = None


def main():
    """Main application entry point"""
    st.set_page_config(**PAGE_CONFIG)

    render_custom_css()
    render_hero_section()

    # Sidebar configuration
    model_choice, max_silence_duration = render_sidebar()

    # Main content layout
    st.markdown("---")

    st.subheader("📹 Video Upload")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=SUPPORTED_VIDEO_FORMATS,
        help="Select a video file to transcribe (max size depends on your system)",
    )

    # Initialize session state
    if SESSION_STATE_RESULTS not in st.session_state:
        st.session_state[SESSION_STATE_RESULTS] = None
    if SESSION_STATE_UPLOAD_KEY not in st.session_state:
        st.session_state[SESSION_STATE_UPLOAD_KEY] = None

    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_size = uploaded_file.size
        logger.info(f"[MAIN] File uploaded: {file_name} (size: {file_size} bytes)")

        # Check if this is a new upload
        current_upload_key = f"{file_name}:{file_size}"
        if st.session_state.get(SESSION_STATE_UPLOAD_KEY) != current_upload_key:
            st.session_state[SESSION_STATE_UPLOAD_KEY] = current_upload_key
            st.session_state[SESSION_STATE_RESULTS] = None
            logger.info("[MAIN] Detected new upload. Cleared previous results.")

        # Display file info
        log_container = st.container()
        with log_container:
            st.info(f"📤 File: **{file_name}** ({file_size / 1024 / 1024:.2f} MB)")

        st.markdown("---")
        st.subheader("🚀 Transcription")

        start_button = st.button("🚀 Start Transcription", type="primary", use_container_width=True)

        if start_button:
            process_video(uploaded_file, model_choice, max_silence_duration)

        # Display results if available
        results = st.session_state.get(SESSION_STATE_RESULTS)
        if results:
            render_results_section(results, uploaded_file)

    else:
        # Landing page when no file is uploaded
        render_landing_page()

    # Footer
    render_footer()


if __name__ == "__main__":
    main()
