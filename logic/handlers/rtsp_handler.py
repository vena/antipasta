"""
RTSP Frame Handler
------------------
Extracts frames from RTSP/RTSPS streams using FFmpeg.
Supports both single-shot polling and persistent background streaming.
Utilizes hardware-accelerated decoding paths when available.
"""

import subprocess
import logging
import os
import time
import signal
import threading
from typing import Optional

import config
from core.state import APP_STATE
from handlers.stream_utils import extract_next_jpeg

logger = logging.getLogger(__name__)

class RTSPHandler:
    def __init__(self, rtsp_url: str, persistent: bool = False):
        self.rtsp_url = rtsp_url
        self.persistent = persistent
        
        self._lock = threading.Lock()
        self._latest_frame: Optional[bytes] = None
        
        if self.persistent:
            logger.info(f"Starting persistent RTSP connection to {self.rtsp_url}...")
            self._worker_thread = threading.Thread(target=self._persistent_worker, daemon=True)
            self._worker_thread.start()
        
    def _get_ffmpeg_args(self, is_persistent: bool) -> list:
        """
        Generates platform-specific FFmpeg arguments based on the 
        active hardware acceleration mode.
        """
        hw = APP_STATE.get_hardware_report().logic
        
        # We use -loglevel error so FFmpeg only outputs text when it is dying.
        args = ['ffmpeg', '-hide_banner', '-loglevel', 'error']
        
        # Hardware Input/Decoder Selection
        if hw.mode == "vaapi":
            args.extend([
                '-hwaccel', 'vaapi',
                '-hwaccel_device', hw.device,
                '-hwaccel_output_format', 'vaapi',
            ])
            
        # -timeout (in microseconds) protects against silent stream hangs.
        # If FFmpeg receives no data for 10 seconds, it will exit, allowing 
        # our Python process to catch it and restart cleanly.
        args.extend([
            '-timeout', '10000000', 
            '-rtsp_transport', 'tcp'
        ])

        # Handle TLS for RTSPS
        if self.rtsp_url.startswith('rtsps'):
            args.extend(['-tls_verify', '1' if config.STRICT_TLS else '0'])
            if config.CUSTOM_CA_CERT_PATH and os.path.exists(config.CUSTOM_CA_CERT_PATH):
                args.extend(['-ca_file', config.CUSTOM_CA_CERT_PATH])

        # Input and Decoder
        if hw.mode == "v4l2":
            args.extend(['-c:v', 'h264_v4l2m2m', '-i', self.rtsp_url])
        else:
            args.extend(['-i', self.rtsp_url])

        # Filter Chain
        if hw.mode == "vaapi":
            args.extend(['-vf', 'hwdownload,format=nv12'])

        # Output Configuration
        if is_persistent:
            # -r 5 drops the framerate to 5 FPS to dramatically reduce CPU usage.
            # -f mpjpeg wraps the stream in HTTP boundaries to prevent EXIF corruption.
            args.extend([
                '-r', '5',
                '-f', 'mpjpeg',
                '-vcodec', 'mjpeg',
                'pipe:1'
            ])
        else:
            args.extend([
                '-f', 'image2',
                '-vframes', '1',
                '-update', '1',
                'pipe:1'
            ])
        
        return args

    def _persistent_worker(self):
        """
        Background daemon that continuously pulls frames from the RTSP stream.
        """
        while True:
            cmd = self._get_ffmpeg_args(is_persistent=True)
            process = None
            
            try:
                # We route stderr to a PIPE so we can capture crash logs.
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, 
                    preexec_fn=os.setsid
                )
                
                logger.info("RTSP persistent stream established via FFmpeg.")
                raw_buffer = bytearray()
                
                while True:
                    chunk = process.stdout.read(16384)
                    if not chunk:
                        # If stdout closes, the process is dead. Pull the crash log.
                        err_log = process.stderr.read().decode('utf-8', errors='ignore').strip()
                        if err_log:
                            logger.error(f"FFmpeg Crash Log: {err_log}")
                            
                        logger.error("RTSP FFmpeg process closed stdout. Restarting...")
                        break
                        
                    raw_buffer.extend(chunk)
                    
                    # Exhaust all complete frames currently in the buffer
                    while True:
                        frame, new_buffer = extract_next_jpeg(raw_buffer)
                        if frame:
                            with self._lock:
                                self._latest_frame = frame
                            raw_buffer = new_buffer
                        else:
                            break
                    
                    # Memory Safety Check
                    if len(raw_buffer) > config.MAX_FRAME_SIZE_BYTES:
                        logger.error(f"RTSP buffer exceeded {config.MAX_FRAME_SIZE_BYTES} bytes without finding a frame. Stream corrupted, restarting...")
                        break
                        
            except Exception as e:
                logger.warning(f"Persistent RTSP stream encountered an error: {e}")
            finally:
                if process:
                    try:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                
                with self._lock:
                    self._latest_frame = None
            
            time.sleep(5)

    def get_frame(self) -> Optional[bytes]:
        """
        Returns the latest frame. If persistent mode is active, it performs 
        a near-instant RAM read. Otherwise, it invokes a blocking FFmpeg capture.
        """
        if self.persistent:
            with self._lock:
                return self._latest_frame

        # --- Legacy One-Shot Polling Mode ---
        cmd = self._get_ffmpeg_args(is_persistent=False)
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode == 0:
                return result.stdout
            else:
                error_msg = result.stderr.decode('utf-8', errors='ignore').strip()
                logger.warning(f"FFmpeg capture failed (Code {result.returncode}): {error_msg}")
        except subprocess.TimeoutExpired:
            logger.warning("FFmpeg capture timed out (10s limit).")
        except Exception as e:
            logger.warning(f"FFmpeg capture exception: {e}")
            
        return None