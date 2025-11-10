"""
Transcription module for Chunkformer model
"""

import io
import logging
import time
from contextlib import redirect_stderr, redirect_stdout
from typing import Dict, List, Tuple

import streamlit as st
from utils import timestamp_to_seconds

logger = logging.getLogger(__name__)


@st.cache_resource
def load_model(model_path: str = "chunkformer-ctc-large-vie"):
    """Load and cache the Chunkformer model

    Args:
        model_path: Path or HuggingFace ID of the model

    Returns:
        Loaded ChunkFormer model or None if failed
    """
    try:
        # Import here to avoid circular dependencies
        from chunkformer.chunkformer_model import ChunkFormerModel

        logger.info(f"[LOAD_MODEL] Starting to load model: {model_path}")
        start_time = time.time()

        st.write(f"Loading model: {model_path}")
        model = ChunkFormerModel.from_pretrained(model_path)
        model.eval()

        elapsed = time.time() - start_time
        logger.info(f"[LOAD_MODEL] Model loaded successfully in {elapsed:.2f}s")
        return model
    except Exception as e:
        logger.error(f"[LOAD_MODEL] Failed to load model: {e}", exc_info=True)
        st.error(f"Failed to load model: {e}")
        return None


def transcribe_audio(
    model,
    media_path: str,
    chunk_size: int = 64,
    left_context_size: int = 128,
    right_context_size: int = 128,
    total_batch_duration: int = 1800,
    max_silence_duration: float = 0.5,
) -> Tuple[List[Dict], str]:
    """Transcribe audio/video using Chunkformer model's endless_decode

    Accepts both audio files (.wav, .mp3, etc.) and video files (.mp4, .mkv, etc.)

    Args:
        model: Loaded Chunkformer model
        media_path: Path to audio or video file
        chunk_size: Size of chunks for processing
        left_context_size: Left context window size
        right_context_size: Right context window size
        total_batch_duration: Total batch duration in seconds
        max_silence_duration: Maximum silence duration in seconds for sentence break detection

    Returns:
        Tuple of (segments list, full transcript string)

    The endless_decode method returns segments with timing information in format:
    [{'start': time_in_seconds, 'end': time_in_seconds, 'decode': text}, ...]
    """
    try:
        logger.info(f"[TRANSCRIBE] Starting transcription of: {media_path}")
        overall_start = time.time()

        with st.spinner("Running Chunkformer endless_decode..."):
            status_text = st.empty()

            # Use endless_decode to get results with timestamps
            logger.info("[TRANSCRIBE] Calling model.endless_decode()...")
            decode_start = time.time()

            # Suppress stderr/stdout to avoid tqdm broken pipe errors
            with redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
                decode_result = model.endless_decode(
                    audio_path=media_path,  # Can be audio or video file
                    chunk_size=chunk_size,
                    left_context_size=left_context_size,
                    right_context_size=right_context_size,
                    total_batch_duration=total_batch_duration,
                    return_timestamps=True,
                    max_silence_duration=max_silence_duration,
                )

            decode_elapsed = time.time() - decode_start
            logger.info(f"[TRANSCRIBE] endless_decode completed in {decode_elapsed:.2f}s")

            status_text.text("Processing transcription results...")
            num_segments = len(decode_result) if isinstance(decode_result, list) else "unknown"
            logger.info(f"[TRANSCRIBE] Processing {num_segments} segments...")

            # Convert decode_result to segments format
            segments = []
            full_transcript = ""

            if isinstance(decode_result, list):
                logger.info(
                    f"[TRANSCRIBE] decode_result is list with " f"{len(decode_result)} items"
                )
                for idx, item in enumerate(decode_result):
                    if isinstance(item, dict):
                        # Item format:
                        # {'start': 'hh:mm:ss:ms', 'end': 'hh:mm:ss:ms',
                        #  'decode': str}
                        start_str = str(item.get("start", "00:00:00:000"))
                        end_str = str(item.get("end", "00:00:00:000"))

                        start_float = timestamp_to_seconds(start_str)
                        end_float = timestamp_to_seconds(end_str)

                        segment = {
                            "start": start_float,
                            "end": end_float,
                            "text": item.get("decode", ""),
                            "confidence": 1.0,
                        }
                        segments.append(segment)
                        full_transcript += item.get("decode", "") + " "
                        if idx < 3:  # Log first 3 segments
                            logger.info(
                                f"[TRANSCRIBE] Segment {idx}: "
                                f"[{start_str} ({start_float:.2f}s) - "
                                f"{end_str} ({end_float:.2f}s)] "
                                f"{segment['text'][:50]}"
                            )
            else:
                logger.warning(
                    f"[TRANSCRIBE] decode_result is not a list, " f"type: {type(decode_result)}"
                )

            status_text.empty()

            overall_elapsed = time.time() - overall_start
            logger.info(
                f"[TRANSCRIBE] Transcription complete! "
                f"{len(segments)} segments in {overall_elapsed:.2f}s"
            )

            return segments, full_transcript.strip()

    except Exception as e:
        logger.error(f"[TRANSCRIBE] Error during transcription: {e}", exc_info=True)
        st.error(f"Error during transcription: {e}")
        import traceback

        st.error(f"Traceback: {traceback.format_exc()}")
        return [], ""
