from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CLIPFORGE_",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "CLIPForge"
    version: str = "0.3.0"
    environment: str = "development"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    api_keys: list[str] = Field(default_factory=list)
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8000"])

    model_backend: str = "mock"
    model_name: str = "ViT-B-32"
    model_pretrained: str = "laion2b_s34b_b79k"
    model_id: str = "google/siglip2-base-patch16-224"
    model_device: str = "auto"
    model_precision: str = "auto"
    model_compile: bool = False
    model_attention: str = "sdpa"
    model_warmup: bool = True
    embedding_dimension: int = 512
    max_batch_size: int = 64
    max_image_bytes: int = 10 * 1024 * 1024

    request_timeout_seconds: float = 30.0
    rate_limit_per_minute: int = 120
    index_capacity: int = 100_000
    vector_store_path: str = "data/clipforge.db"
    default_tenant_id: str = "default"
    default_collection: str = "main"
    job_workers: int = 1
    job_history_limit: int = 1000
    max_concurrent_inference: int = 1
    inference_cache_size: int = 4096

    @field_validator("model_backend")
    @classmethod
    def validate_backend(cls, value: str) -> str:
        value = value.lower()
        if value not in {"mock", "openclip", "siglip2"}:
            raise ValueError("model_backend must be 'mock', 'openclip' or 'siglip2'")
        return value

    @field_validator("model_precision")
    @classmethod
    def validate_precision(cls, value: str) -> str:
        value = value.lower()
        if value not in {"auto", "fp32", "fp16", "bf16"}:
            raise ValueError("model_precision must be auto, fp32, fp16 or bf16")
        return value

    @field_validator("model_attention")
    @classmethod
    def validate_attention(cls, value: str) -> str:
        value = value.lower()
        if value not in {"auto", "eager", "sdpa", "flash_attention_2"}:
            raise ValueError("Unsupported model_attention implementation")
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()
