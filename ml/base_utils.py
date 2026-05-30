"""
ML Base Utilities
-----------------
Handles ETag-based model synchronization with real-time status reporting.
Ensures the ML Engine can report download progress to the Logic controller.
"""

import os
import logging
import requests
import shutil

logger = logging.getLogger("AntiPasta.MLUtils")

def setup_model_directory(model_dir: str, model_url: str, pt_filename: str, status_callback=None):
    """
    Handles ETag-based model synchronization.
    
    Args:
        model_dir: Local path to store weights.
        model_url: Upstream URL for the model file.
        pt_filename: Expected local filename.
        status_callback: Optional function to update the global engine status message.
    """
    os.makedirs(model_dir, exist_ok=True)
    target_path = os.path.join(model_dir, pt_filename)
    etag_path = os.path.join(model_dir, f"{pt_filename}.etag")

    def update_status(msg):
        if status_callback:
            status_callback(msg)
        logger.info(msg)

    # Fetch remote ETag
    remote_etag = None
    try:
        head_resp = requests.head(model_url, allow_redirects=True, timeout=10)
        remote_etag = head_resp.headers.get('ETag', '').strip('"')
    except Exception as e:
        logger.warning(f"Could not fetch remote ETag for {model_url}: {e}")

    # Check local ETag
    local_etag = None
    if os.path.exists(etag_path):
        with open(etag_path, 'r') as f:
            local_etag = f.read().strip()

    # Determine if download is necessary
    should_download = not os.path.exists(target_path) or (remote_etag and remote_etag != local_etag)

    if should_download:
        update_status(f"Downloading model weights from {model_url}...")
        try:
            with requests.get(model_url, stream=True, allow_redirects=True) as r:
                r.raise_for_status()
                total_size = int(r.headers.get('content-length', 0))
                downloaded = 0
                
                with open(target_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0 and downloaded % (1024 * 1024) == 0:
                                percent = int((downloaded / total_size) * 100)
                                update_status(f"Downloading weights: {percent}%")
            
            if remote_etag:
                with open(etag_path, 'w') as f:
                    f.write(remote_etag)
            update_status("Download complete.")
            return True # Indicates a change occurred
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            if not os.path.exists(target_path):
                raise RuntimeError("No local model found and download failed.") from e
    
    return False # No update performed