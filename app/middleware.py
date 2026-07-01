import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("clipforge.http")


class RequestContextMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: object, timeout_seconds: float) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.timeout_seconds = timeout_seconds

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        started = time.perf_counter()
        try:
            async with asyncio.timeout(self.timeout_seconds):
                response = await call_next(request)
        except TimeoutError:
            response = JSONResponse(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                content={"detail": {"code": "timeout", "message": "Request timed out."}},
            )
        duration_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time-Ms"] = f"{duration_ms:.2f}"
        logger.info(
            "request_complete",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
            },
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: object, requests_per_minute: int) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self.limit = requests_per_minute
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path.startswith(("/health", "/assets")) or self.limit <= 0:
            return await call_next(request)
        identity = request.headers.get("X-API-Key") or (
            request.client.host if request.client else "unknown"
        )
        now = time.monotonic()
        async with self._lock:
            window = self._requests[identity]
            while window and window[0] <= now - 60:
                window.popleft()
            if len(window) >= self.limit:
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": {
                            "code": "rate_limit_exceeded",
                            "message": "Too many requests. Try again shortly.",
                        }
                    },
                    headers={"Retry-After": "60"},
                )
            window.append(now)
        return await call_next(request)
