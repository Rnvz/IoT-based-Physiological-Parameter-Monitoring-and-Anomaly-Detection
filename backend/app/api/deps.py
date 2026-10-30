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

def verify_read_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    Enforces X-API-Key on read (GET) endpoints only when READ_AUTH_REQUIRED=true.
    When READ_AUTH_REQUIRED=false (default), read endpoints are publicly accessible.
    """
    if not settings.read_auth_required:
        return x_api_key
    return verify_api_key(x_api_key)

def get_client_ip(request: Request) -> str:
    """
    Extracts client IP address.
    ONLY trusts CF-Connecting-IP if the direct socket peer is loopback (127.0.0.1, ::1, or testclient).
    Otherwise, uses the direct socket peer IP to prevent IP spoofing by untrusted clients.
    """
    peer_ip = request.client.host if request.client else "unknown"
    if peer_ip in ("127.0.0.1", "::1", "localhost", "testclient"):
        cf_ip = request.headers.get("CF-Connecting-IP") or request.headers.get("cf-connecting-ip")
        if cf_ip:
            return cf_ip.strip()
    return peer_ip

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
        client_ip = get_client_ip(request)
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
