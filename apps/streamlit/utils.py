"""
Utility functions for the Chunkformer Streamlit app
"""

import logging
import mimetypes
from typing import Dict, List

logger = logging.getLogger(__name__)


def timestamp_to_seconds(timestamp_str: str) -> float:
    """Convert hh:mm:ss:ms format to seconds

    Args:
        timestamp_str: Timestamp in format '00:00:05:123'

    Returns:
        Total seconds as float (e.g., 5.123)
    """
    try:
        parts = str(timestamp_str).split(":")
        if len(parts) == 4:
            hours, minutes, seconds, milliseconds = map(int, parts)
            total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
            return total_seconds
        else:
            # Try parsing as float directly
            return float(timestamp_str)
    except (ValueError, AttributeError, TypeError):
        logger.warning(f"Could not parse timestamp: {timestamp_str}, using 0.0")
        return 0.0


def format_timestamp(seconds: float) -> str:
    """Format seconds to MM:SS format

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string in MM:SS format
    """
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    return f"{minutes:02d}:{secs:02d}"


def create_subtitle_srt(segments: List[Dict]) -> str:
    """Create SRT format subtitles

    Args:
        segments: List of segment dictionaries with start, end, and text

    Returns:
        SRT formatted subtitle string
    """
    srt_content = ""
    for i, segment in enumerate(segments, 1):
        start = format_timestamp(segment["start"])
        end = format_timestamp(segment["end"])
        srt_content += f"{i}\n{start} --> {end}\n{segment['text']}\n\n"
    return srt_content


def guess_video_mime_type(file_name: str) -> str:
    """Best-effort MIME type detection for video assets

    Args:
        file_name: Name of the video file

    Returns:
        MIME type string (defaults to video/mp4)
    """
    if not file_name:
        return "video/mp4"
    mime_type, _ = mimetypes.guess_type(file_name)
    if mime_type and mime_type.startswith("video/"):
        return mime_type
    return "video/mp4"


def get_transcript_at_time(
    segments: List[Dict], current_time: float, context_window: float = 5.0
) -> str:
    """Get transcript text for current playback time with context

    Args:
        segments: List of transcript segments
        current_time: Current playback time in seconds
        context_window: Time window in seconds for context

    Returns:
        Formatted transcript text with current segment highlighted
    """
    transcript_lines = []

    for segment in segments:
        if segment["start"] - context_window <= current_time <= segment["end"] + context_window:
            # Highlight current segment
            if segment["start"] <= current_time <= segment["end"]:
                transcript_lines.append(f"► {segment['text']} ◄")
            else:
                transcript_lines.append(segment["text"])

    return " ".join(transcript_lines) if transcript_lines else "Loading transcript..."


def prepare_segments_for_player(segments: List[Dict]) -> List[Dict]:
    """Normalize segment payload for the synchronized player

    Args:
        segments: Raw segment data from transcription

    Returns:
        Normalized segments with index, start, end, and text
    """
    prepared_segments: List[Dict] = []
    for idx, segment in enumerate(segments, start=1):
        start = float(segment.get("start", 0.0) or 0.0)
        end = float(segment.get("end", start) or start)
        if end <= start:
            end = start + 0.01  # ensure strictly increasing to avoid zero-length highlights
        prepared_segments.append(
            {
                "index": idx,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": segment.get("text", ""),
            }
        )
    return prepared_segments
