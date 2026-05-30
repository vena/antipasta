"""
Remote Image Handler
--------------------
Provides a unified interface for fetching remote images for the /test/ API endpoint.
Handles Server-Side Request Forgery (SSRF) protection, URL normalization, and 
memory-safe stream limits.
"""

import urllib.parse
import socket
import ipaddress
import requests
import logging
from typing import Optional
import config

logger = logging.getLogger(__name__)

class RemoteImageHandler:
    def __init__(self, target_url: str):
        self.target_url = self._repair_url(target_url)

    def _repair_url(self, url: str) -> str:
        """
        Defensive URL repairing for reverse proxies.
        Proxies like Nginx or Traefik often strip consecutive slashes in the path,
        mutating "http://domain" into "http:/domain". This breaks urllib.parse.
        """
        if url.startswith('http:/') and not url.startswith('http://'):
            return url.replace('http:/', 'http://', 1)
        elif url.startswith('https:/') and not url.startswith('https://'):
            return url.replace('https:/', 'https://', 1)
        return url

    def _is_safe_url(self) -> bool:
        """
        Validates URLs to prevent recursive SSRF against the container itself.
        Note: This is a stripped-down check. We rely on the ALLOW_TEST_API flag 
        in the routing layer to ensure users intentionally opt-in to network exposure.
        """
        try:
            parsed = urllib.parse.urlparse(self.target_url)
            if parsed.scheme not in ('http', 'https'):
                return False
                
            hostname = parsed.hostname
            if not hostname:
                return False
                
            ip = socket.gethostbyname(hostname)
            ip_obj = ipaddress.ip_address(ip)
            
            # Block internal loopbacks and unspecified addresses (e.g., 0.0.0.0) 
            # to prevent the container from infinitely proxying itself and causing 
            # a crash or out-of-memory exception.
            if ip_obj.is_loopback or ip_obj.is_unspecified:
                return False
                
            return True
        except Exception as e:
            logger.warning(f"URL security validation failed: {e}")
            return False

    def get_frame(self) -> Optional[bytes]:
        """
        Fetches the image into memory with strict bounds checking.
        Raises ValueError for security or memory violations.
        """
        if not self._is_safe_url():
            raise ValueError("Invalid or restricted URL.")
            
        # 10MB streaming size limit to prevent Out-Of-Memory (OOM) 
        # vulnerabilities if a user supplies a URL to an infinitely large file.
        max_size = 10 * 1024 * 1024  
        
        # Identify our requests so remote servers have context
        headers = {
            "User-Agent": f"AntiPasta/{config.VERSION} (Testing Endpoint)"
        }
        
        try:
            with requests.get(self.target_url, stream=True, timeout=10, headers=headers) as r:
                r.raise_for_status()
                
                # Fast rejection if Content-Length header provides the size ahead of time
                if int(r.headers.get('Content-Length', 0)) > max_size:
                    raise ValueError("Payload exceeds 10MB limit.")
                
                fetched_bytes = bytearray()
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        fetched_bytes.extend(chunk)
                        if len(fetched_bytes) > max_size:
                            raise ValueError("Payload exceeds 10MB limit.")
                            
                return bytes(fetched_bytes)
                
        except ValueError:
            raise
        except Exception as e:
            logger.warning(f"Remote fetch failed: {e}")
            return None