import time
import hmac
from collections import defaultdict, deque
from typing import Optional
from fastapi import Header, HTTPException, Request, status
from app.config import settings

def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    Validates X-API-Key header using constant-time comparison.
    Refuses if API_KEY is empty or mismatch occurs.
    """
    configured_key = settings.api_key.strip() if settings.api_key else ""
    if not configured_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY is not configured on server"
        )

    if not x_api_key or not hmac.compare_digest(x_api_key.strip(), configured_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return x_api_key

class InMemoryRateLimiter:
    """
    Sliding window in-memory rate limiter per client IP.
    Default: 20 requests per 10-second window.
    Zero external dependencies, minimal memory footprint.
    """
    def __init__(self, max_requests: int = 20, window_seconds: float = 10.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = defaultdict(deque)

    def check_rate_limit(self, request: Request):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()
        window = self.requests[client_ip]

        # Evict timestamps older than window
        while window and now - window[0] > self.window_seconds:
            window.popleft()

        if len(window) >= self.max_requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please wait."
            )

        window.append(now)

rate_limiter = InMemoryRateLimiter(max_requests=20, window_seconds=10.0)

def rate_limit_post(request: Request):
    rate_limiter.check_rate_limit(request)
