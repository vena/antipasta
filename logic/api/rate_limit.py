"""
API Rate Limiter
----------------
Provides a thread-safe Token Bucket rate limiting decorator for Flask routes.
Protects the AI engine and the camera connection pool from being overwhelmed 
by aggressive polling from clients or home automation dashboards.
"""

import time
from threading import Lock
from functools import wraps
from flask import request, jsonify

class TokenBucket:
    def __init__(self, rate: float, per_seconds: float):
        self.rate = float(rate)
        self.per_seconds = float(per_seconds)
        self.tokens = self.rate
        self.last_update = time.monotonic()
        self.lock = Lock()

    def consume(self) -> bool:
        """
        Attempts to consume a single token. Replenishes the bucket based on 
        the time elapsed since the last check.
        Returns True if a token was available, False otherwise.
        """
        with self.lock:
            now = time.monotonic()
            time_passed = now - self.last_update
            
            # Replenish tokens
            self.tokens += time_passed * (self.rate / self.per_seconds)
            if self.tokens > self.rate:
                self.tokens = self.rate
            self.last_update = now

            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return True
            return False

class RateLimiter:
    def __init__(self):
        self.buckets = {}
        self.lock = Lock()

    def get_bucket(self, ip: str, rate: float, per_seconds: float) -> TokenBucket:
        key = f"{ip}:{rate}:{per_seconds}"
        with self.lock:
            if key not in self.buckets:
                self.buckets[key] = TokenBucket(rate, per_seconds)
            return self.buckets[key]

# Singleton instance to persist buckets across requests
_limiter = RateLimiter()

def rate_limit(rate: float, per_seconds: float):
    """
    Decorator to limit API route access using a Token Bucket.
    Args:
        rate: Maximum burst capacity (number of requests).
        per_seconds: The time window over which the tokens replenish.
    """
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            ip = request.remote_addr or "127.0.0.1"
            bucket = _limiter.get_bucket(ip, rate, per_seconds)
            
            if not bucket.consume():
                return jsonify({"error": "Too Many Requests. Rate limit exceeded."}), 429
                
            return f(*args, **kwargs)
        return wrapped
    return decorator