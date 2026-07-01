import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import make_asgi_app

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.middleware import RateLimitMiddleware, RequestContextMiddleware
from app.services.runtime import create_runtime

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("clipforge")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    logger.info("loading_model backend=%s", settings.model_backend)
    app.state.runtime = create_runtime(settings)
    logger.info("service_ready model=%s", app.state.runtime.model.name)
    yield
    app.state.runtime.jobs.close()
    app.state.runtime.model.close()
    app.state.runtime.store.close()
    logger.info("service_stopped")


app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    description="Multimodal embeddings, similarity, and semantic search.",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
app.add_middleware(
    RequestContextMiddleware,
    timeout_seconds=settings.request_timeout_seconds,
)
app.add_middleware(RateLimitMiddleware, requests_per_minute=settings.rate_limit_per_minute)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception path=%s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": {"code": "internal_error", "message": "Internal server error."}},
    )


app.include_router(router, prefix="/api/v1")
app.mount("/metrics", make_asgi_app())

web_dir = Path(__file__).parent / "web"
app.mount("/assets", StaticFiles(directory=web_dir / "assets"), name="assets")


@app.get("/", include_in_schema=False)
async def console() -> FileResponse:
    return FileResponse(web_dir / "index.html")


@app.get("/{path:path}", include_in_schema=False)
async def spa_fallback(path: str) -> FileResponse:
    return FileResponse(web_dir / "index.html")
