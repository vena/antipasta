"""
Module to handle frame extraction from the Bambu Lab proprietary Chamber Image protocol.
Uses a robust marker-based approach over TLS to extract continuous MJPEG streams, 
ignoring proprietary headers to maximize compatibility across firmware versions.
"""
import socket
import ssl
import struct
import time
import os
import logging
import threading
from typing import Optional

import config
from handlers.stream_utils import extract_next_jpeg

logger = logging.getLogger(__name__)

class ChamberImageHandler:
    def __init__(self, ip: str, access_code: str, persistent: bool = False):
        self.ip = ip
        self.access_code = access_code
        self.port = 6000
        self.persistent = persistent
        
        self._lock = threading.Lock()
        self._latest_frame: Optional[bytes] = None
        
        if self.persistent:
            logger.info(f"Starting persistent Chamber Image connection to {self.ip}...")
            self._worker_thread = threading.Thread(target=self._persistent_worker, daemon=True)
            self._worker_thread.start()

    def create_auth_payload(self) -> bytes:
        """
        Constructs the binary authentication packet required by the printer's 
        internal camera service.
        """
        username = b"bblp"
        access_code_bytes = self.access_code.encode("utf-8")
        
        return struct.pack(
            "<II8s32s32s",
            0x40,           # Magic identifier for the camera service packet
            0x3000,         # Command: Request stream start
            b"\x00" * 8,    # Padding/Reserved
            username.ljust(32, b"\x00"),
            access_code_bytes.ljust(32, b"\x00")
        )

    def _create_ssl_context(self) -> ssl.SSLContext:
        """Prepares the TLS context required to securely connect to the printer."""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        
        if config.STRICT_TLS:
            context.verify_mode = ssl.CERT_REQUIRED
            context.check_hostname = True
        else:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE

        if config.CUSTOM_CA_CERT_PATH and os.path.exists(config.CUSTOM_CA_CERT_PATH):
            context.load_verify_locations(config.CUSTOM_CA_CERT_PATH)
            
        return context

    def _persistent_worker(self):
        """
        Background daemon that continuously pulls frames from the printer.
        Maintains connection state and isolates latency from the main application loop.
        """
        sni_hostname = config.PRINTER_SERIAL
        context = self._create_ssl_context()
        
        while True:
            ssock = None
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(10) # 10s socket timeout prevents hanging on network drops
                
                ssock = context.wrap_socket(sock, server_hostname=sni_hostname)
                ssock.connect((self.ip, self.port))
                ssock.sendall(self.create_auth_payload())
                
                logger.info("Chamber Image persistent stream established.")
                
                raw_buffer = bytearray()
                
                # Continuously read frames as fast as the printer sends them
                while True:
                    chunk = ssock.recv(16384)
                    if not chunk:
                        logger.error("Chamber Image socket closed remotely. Reconnecting...")
                        break
                        
                    raw_buffer.extend(chunk)
                    
                    # Exhaust all complete frames currently in the buffer.
                    # This ensures _latest_frame is truly the *latest* available,
                    # preventing the buffer from accumulating stale frames if 
                    # multiple frames arrive in a single socket chunk.
                    while True:
                        frame, new_buffer = extract_next_jpeg(raw_buffer)
                        if frame:
                            with self._lock:
                                self._latest_frame = frame
                            raw_buffer = new_buffer
                        else:
                            break
                    
                    # Memory Safety Check: Guarantee the buffer never leaks
                    if len(raw_buffer) > config.MAX_FRAME_SIZE_BYTES:
                        logger.error(f"Chamber Image buffer exceeded {config.MAX_FRAME_SIZE_BYTES} bytes without finding a frame. Stream corrupted, reconnecting...")
                        break
                        
            except Exception as e:
                logger.warning(f"Persistent Chamber Image stream disconnected: {e}")
            finally:
                if ssock:
                    ssock.close()
                    
                # Prevent the application from analyzing stale frames while reconnecting
                with self._lock:
                    self._latest_frame = None
            
            # Backoff before reconnecting to prevent log spam if printer is powered off
            time.sleep(5)

    def get_frame(self) -> Optional[bytes]:
        """
        Returns the latest frame. If persistent mode is active, it performs 
        a near-instant RAM read. Otherwise, it initiates a blocking one-shot capture.
        """
        if self.persistent:
            with self._lock:
                return self._latest_frame

        # --- Legacy One-Shot Polling Mode ---
        context = self._create_ssl_context()
        sni_hostname = config.PRINTER_SERIAL
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        ssock = None
        
        try:
            ssock = context.wrap_socket(sock, server_hostname=sni_hostname)
            ssock.connect((self.ip, self.port))
            ssock.sendall(self.create_auth_payload())
            
            raw_buffer = bytearray()
            start_time = time.time()
            
            # Buffer until we find standard JPEG SOI and EOI markers
            while time.time() - start_time < 10:
                chunk = ssock.recv(16384)
                if not chunk: break
                raw_buffer.extend(chunk)
                
                frame, raw_buffer = extract_next_jpeg(raw_buffer)
                if frame:
                    return frame
                
                if len(raw_buffer) > config.MAX_FRAME_SIZE_BYTES:
                    logger.error(f"Chamber Image polling exceeded {config.MAX_FRAME_SIZE_BYTES} bytes. Stream likely corrupted.")
                    return None
                    
        except Exception as e:
            logger.warning(f"Chamber Image polling capture failed: {e}")
        finally:
            if ssock:
                ssock.close()
            else:
                sock.close()
                
        return None