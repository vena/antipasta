"""
Hardware Discovery Service
--------------------------
Performs literal probes of the container environment to identify available 
video decoding acceleration. This avoids "magic" logic based solely on 
Docker architecture variables.
"""

import os
import platform
import logging
import subprocess
from core.models import HardwareStats

logger = logging.getLogger("AntiPasta.Core.Hardware")

class HardwareProbe:
    @staticmethod
    def probe_logic_acceleration(requested_mode: str) -> HardwareStats:
        """
        Determines the effective video decoding mode by checking for 
        device nodes and verifying driver compatibility.
        """
        arch = platform.machine().lower()
        
        # User explicitly requested CPU or is on an unsupported architecture
        if requested_mode == "cpu":
            return HardwareStats(mode="cpu", device="CPU", active=True, note="User forced CPU mode")

        # Check for Intel/AMD VAAPI (AMD64 Linux)
        # /dev/dri/renderD128 is the standard entry point for headless GPU acceleration.
        vaapi_device = "/dev/dri/renderD128"
        if (requested_mode in ["auto", "vaapi"]) and os.path.exists(vaapi_device):
            try:
                # To verify the driver, we must run a synthetic transcode.
                # -f lavfi -i nullsrc: Generates a single empty 1x1 pixel in memory.
                # -init_hw_device: Initializes the hardware context.
                # -filter_hw_device: Binds the filter graph to that context.
                # -f null -: Discards the output.
                cmd = [
                    "ffmpeg", "-hide_banner",
                    "-f", "lavfi", "-i", "nullsrc",
                    "-init_hw_device", f"vaapi=gpu:{vaapi_device}",
                    "-filter_hw_device", "gpu",
                    "-frames:v", "1",
                    "-f", "null", "-"
                ]
                result = subprocess.run(cmd, capture_output=True, timeout=5)
                
                if result.returncode == 0:
                    return HardwareStats(
                        mode="vaapi", 
                        device=vaapi_device, 
                        active=True, 
                        note="Intel/AMD VAAPI hardware acceleration verified"
                    )
                else:
                    # Capturing the actual stderr allows us to see if it's a permission 
                    # issue or a missing driver (e.g. 'Failed to open DRM render node')
                    error_out = result.stderr.decode().strip().split('\n')[-1]
                    logger.warning(f"VAAPI device found but driver failed: {error_out}")
            except Exception as e:
                logger.debug(f"VAAPI probe exception: {e}")

        # Check for Raspberry Pi V4L2 (ARM64 Linux)
        # Note: Pi 5 does NOT have H.264/MJPEG hardware decoders. 
        # We only enable this for Pi 4 and earlier ARM devices that support v4l2m2m.
        v4l2_device = "/dev/video10"
        is_pi = "arm" in arch or "aarch64" in arch
        if (requested_mode in ["auto", "v4l2"]) and is_pi and os.path.exists(v4l2_device):
            # Check if the specific h264_v4l2m2m decoder is available in this ffmpeg build
            try:
                check = subprocess.run(["ffmpeg", "-decoders"], capture_output=True, text=True)
                if "h264_v4l2m2m" in check.stdout:
                    return HardwareStats(
                        mode="v4l2", 
                        device=v4l2_device, 
                        active=True, 
                        note="ARM V4L2 hardware acceleration detected"
                    )
                else:
                    logger.info("V4L2 device found but h264_v4l2m2m decoder is missing (Standard for Pi 5).")
            except Exception:
                pass

        # If auto-detection fails, we fall back to CPU.
        reason = f"No compatible hardware found for {requested_mode} mode"
        if requested_mode != "auto" and requested_mode != "cpu":
            logger.error(f"Requested {requested_mode} acceleration failed. Falling back to CPU.")
            
        return HardwareStats(
            mode="cpu", 
            device="CPU", 
            active=True, 
            note=reason if requested_mode == "auto" else f"Fallback: {reason}"
        )

# Global probe executed at module load or initialization
def get_logic_hardware_report(requested_mode: str) -> HardwareStats:
    return HardwareProbe.probe_logic_acceleration(requested_mode)