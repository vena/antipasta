"""
Stream Utilities
----------------
Shared helper functions for parsing and managing continuous media byte streams.
"""

from typing import Optional, Tuple

def extract_next_jpeg(raw_buffer: bytearray) -> Tuple[Optional[bytes], bytearray]:
    """
    Searches the buffer for standard JPEG Start of Image (SOI) and End of Image (EOI) markers.
    If a complete frame is found, returns the frame and slices the buffer forward to 
    discard the processed frame and any garbage bytes preceding the NEXT frame.
    
    Args:
        raw_buffer: A mutable bytearray containing the raw stream data.
        
    Returns:
        A tuple of (frame_bytes, sliced_buffer). 
        If no frame is complete, returns (None, original_buffer).
    """
    start_idx = raw_buffer.find(b"\xff\xd8")
    if start_idx != -1:
        end_idx = raw_buffer.find(b"\xff\xd9", start_idx)
        if end_idx != -1:
            frame = bytes(raw_buffer[start_idx : end_idx + 2])
            sliced_buffer = raw_buffer[end_idx + 2:]
            return frame, sliced_buffer
            
    return None, raw_buffer