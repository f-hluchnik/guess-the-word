import math
import time

from fastapi import HTTPException, Request, status
from limits import RateLimitItem, parse
from limits.storage import storage_from_string
from limits.strategies import MovingWindowRateLimiter


class RateLimit:
    """Per-client-IP request budget, usable as a FastAPI dependency.

    Behind a proxy, pass the header it puts the real client IP in. Otherwise
    every request appears to come from the proxy and shares one budget.
    Only use a header the proxy overwrites, or clients can spoof it.
    """

    def __init__(
        self, scope: str, limit: str, storage_uri: str, client_ip_header: str | None = None
    ) -> None:
        self._scope = scope
        self._client_ip_header = client_ip_header
        self._item: RateLimitItem = parse(limit)
        self._limiter = MovingWindowRateLimiter(storage_from_string(storage_uri))

    def _client(self, request: Request) -> str:
        if self._client_ip_header and (ip := request.headers.get(self._client_ip_header)):
            return ip
        return request.client.host if request.client else "unknown"

    def __call__(self, request: Request) -> None:
        client = self._client(request)
        if self._limiter.hit(self._item, self._scope, client):
            return
        reset_at = self._limiter.get_window_stats(self._item, self._scope, client).reset_time
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many guesses, slow down.",
            headers={"Retry-After": str(max(1, math.ceil(reset_at - time.time())))},
        )

    def reset(self) -> None:
        self._limiter.storage.reset()
