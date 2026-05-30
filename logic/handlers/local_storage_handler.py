"""
Local Storage Handler
---------------------
Encapsulates all file system operations for image persistence.
Responsible for safely caching live-view frames and archiving failure events.
"""

import os
import io
import time
import logging
from typing import Optional

import config

logger = logging.getLogger("AntiPasta.LocalStorage")

def save_latest_frame(frame_bytes: bytes) -> None:
    """
    Safely updates the diagnostic live-view image using an atomic POSIX replace.
    This guarantees that concurrent web clients never read a half-written file.
    """
    if not frame_bytes:
        return
    tmp_path = f"{config.LAST_FRAME_PATH}.tmp"
    with open(tmp_path, "wb") as f:
        f.write(frame_bytes)
    os.replace(tmp_path, config.LAST_FRAME_PATH)

def archive_failure(run_id: str, annotated_io: io.BytesIO) -> Optional[str]:
    """Saves a permanent record of a failure event for HA notification payloads."""
    if not annotated_io:
        return None
    os.makedirs(config.FAILURES_DIR, exist_ok=True)
    file_path = os.path.join(config.FAILURES_DIR, f"{run_id}.jpg")
    with open(file_path, "wb") as f:
        f.write(annotated_io.getbuffer())
    return file_path

def cleanup_failures():
    """Maintains disk space by enforcing age and count-based retention."""
    if not os.path.exists(config.FAILURES_DIR):
        return
        
    now = time.time()
    retention_sec = config.FAILURE_RETENTION_DAYS * 86400
    files = [os.path.join(config.FAILURES_DIR, f) for f in os.listdir(config.FAILURES_DIR) if f.endswith('.jpg')]
    
    valid_files = []
    for f in files:
        try:
            if (now - os.path.getmtime(f)) > retention_sec:
                os.remove(f)
            else:
                valid_files.append(f)
        except Exception:
            continue

    if len(valid_files) > config.FAILURE_RETENTION_COUNT:
        valid_files.sort(key=os.path.getmtime, reverse=True)
        # Slicing from FAILURE_RETENTION_COUNT onwards gives us the oldest files that exceed our cap
        for f in valid_files[config.FAILURE_RETENTION_COUNT:]:
            try:
                os.remove(f)
            except Exception:
                continue