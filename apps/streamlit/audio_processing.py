"""
Audio and video processing utilities
"""

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


def save_uploaded_file_with_progress(
    uploaded_file,
    destination_path: str,
    progress_bar: Optional[Any] = None,
    status_placeholder: Optional[Any] = None,
    chunk_size: int = 4 * 1024 * 1024,
) -> float:
    """Stream the uploaded file to disk with progress feedback.

    Args:
        uploaded_file: Streamlit uploaded file object
        destination_path: Where to save the file
        progress_bar: Streamlit progress bar component
        status_placeholder: Streamlit placeholder for status text
        chunk_size: Size of chunks to read/write (bytes)

    Returns:
        Elapsed time in seconds
    """
    start_time = time.time()
    total_size = getattr(uploaded_file, "size", None)
    bytes_written = 0
    megabyte = 1024 * 1024

    # Reset file pointer and select the best stream to read from
    base_stream = getattr(uploaded_file, "file", uploaded_file)
    try:
        base_stream.seek(0)
    except Exception:
        pass

    with open(destination_path, "wb") as out_file:
        while True:
            chunk = base_stream.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            bytes_written += len(chunk)

            if total_size and total_size > 0 and progress_bar is not None:
                percent = min(1.0, bytes_written / total_size)
                progress_bar.progress(int(percent * 100))

            if status_placeholder is not None:
                if total_size and total_size > 0:
                    status_placeholder.text(
                        f"Saving… {bytes_written / megabyte:.2f} / {total_size / megabyte:.2f} MB"
                    )
                else:
                    status_placeholder.text(f"Saving… {bytes_written / megabyte:.2f} MB")

    # Ensure UI reflects completion even when file size is unknown
    if progress_bar is not None:
        progress_bar.progress(100)

    elapsed = time.time() - start_time
    if status_placeholder is not None:
        status_placeholder.text(f"Save complete in {elapsed:.2f}s")

    # Reset underlying stream for potential future accesses
    try:
        base_stream.seek(0)
    except Exception:
        pass

    logger.info(
        "[UPLOAD] Saved uploaded file to %s (%.2f MB) in %.2fs",
        destination_path,
        bytes_written / megabyte,
        elapsed,
    )

    return elapsed
