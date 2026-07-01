from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class TextEmbeddingRequest(BaseModel):
    texts: list[str] = Field(min_length=1)

    @field_validator("texts")
    @classmethod
    def reject_blank_text(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("texts cannot contain blank strings")
        return values


class ImageEmbeddingRequest(BaseModel):
    images: list[str] = Field(
        min_length=1,
        description="Base64-encoded images, optionally prefixed with a data URI.",
    )


class EmbeddingResponse(BaseModel):
    model: str
    dimension: int
    embeddings: list[list[float]]
    count: int
    duration_ms: float


class SimilarityRequest(BaseModel):
    texts: list[str] = Field(min_length=1)
    images: list[str] = Field(min_length=1)


class SimilarityResponse(BaseModel):
    model: str
    scores: list[list[float]]
    text_count: int
    image_count: int
    duration_ms: float


class ClassificationRequest(BaseModel):
    images: list[str] = Field(min_length=1)
    labels: list[str] = Field(min_length=2, max_length=1000)
    templates: list[str] = Field(
        default_factory=lambda: [
            "a photo of {}.",
            "a close-up photo of {}.",
            "an image of {}.",
        ],
        min_length=1,
        max_length=32,
    )
    top_k: int = Field(default=5, ge=1, le=100)
    temperature: float = Field(default=0.07, gt=0.001, le=1.0)

    @field_validator("templates")
    @classmethod
    def validate_templates(cls, values: list[str]) -> list[str]:
        if any(template.count("{}") != 1 for template in values):
            raise ValueError("Every prompt template must contain exactly one '{}' placeholder")
        return values


class ClassificationLabel(BaseModel):
    label: str
    score: float
    probability: float


class ClassificationResult(BaseModel):
    image_index: int
    predictions: list[ClassificationLabel]


class ClassificationResponse(BaseModel):
    model: str
    prompt_count: int
    results: list[ClassificationResult]
    duration_ms: float


class IndexTextItem(BaseModel):
    id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexUpsertRequest(BaseModel):
    items: list[IndexTextItem] = Field(min_length=1)


class IndexImageItem(BaseModel):
    id: str = Field(min_length=1, max_length=256)
    image: str = Field(description="Base64 image or image data URI.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexImageRequest(BaseModel):
    items: list[IndexImageItem] = Field(min_length=1)


class IndexDeleteRequest(BaseModel):
    ids: list[str] = Field(min_length=1)


class IndexMutationResponse(BaseModel):
    affected: int
    index_size: int
    collection: str = "main"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    metadata_filter: dict[str, Any] | None = None


class SearchHit(BaseModel):
    id: str
    score: float
    modality: Literal["text", "image"] = "text"
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]
    took_ms: float
    collection: str = "main"


class MultimodalSearchRequest(BaseModel):
    query_type: Literal["text", "image"]
    query: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=100)
    target_modality: Literal["text", "image"] | None = None
    metadata_filter: dict[str, Any] | None = None


class RelevanceFeedback(BaseModel):
    positive_ids: list[str] = Field(default_factory=list, max_length=20)
    negative_ids: list[str] = Field(default_factory=list, max_length=20)
    alpha: float = Field(default=1.0, ge=0.0, le=2.0)
    beta: float = Field(default=0.75, ge=0.0, le=2.0)
    gamma: float = Field(default=0.25, ge=0.0, le=2.0)

    @model_validator(mode="after")
    def validate_feedback(self) -> "RelevanceFeedback":
        overlap = set(self.positive_ids) & set(self.negative_ids)
        if overlap:
            raise ValueError("An item cannot be both positive and negative feedback")
        return self


class InteractiveSearchRequest(MultimodalSearchRequest):
    uncertainty_temperature: float = Field(default=0.07, gt=0.001, le=1.0)
    margin_threshold: float = Field(default=0.12, gt=0.0, lt=1.0)
    entropy_threshold: float = Field(default=0.78, gt=0.0, lt=1.0)
    feedback: RelevanceFeedback = Field(default_factory=RelevanceFeedback)


class UncertaintyResponse(BaseModel):
    margin: float
    normalized_entropy: float
    confidence: float
    needs_clarification: bool
    reason: str


class ClarificationOption(BaseModel):
    id: str
    label: str
    modality: Literal["text", "image"]
    score: float


class ClarificationPrompt(BaseModel):
    question: str
    options: list[ClarificationOption]


class InteractiveSearchResponse(SearchResponse):
    uncertainty: UncertaintyResponse
    clarification: ClarificationPrompt | None = None
    feedback_applied: bool
    query_drift: float


class CollectionCreateRequest(BaseModel):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class CollectionResponse(BaseModel):
    name: str
    tenant_id: str
    dimension: int
    size: int
    created_at: str


class CollectionListResponse(BaseModel):
    collections: list[CollectionResponse]
    total: int


class BatchIndexJobRequest(BaseModel):
    collection: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
    items: list[IndexTextItem] = Field(min_length=1, max_length=10_000)


class JobResponse(BaseModel):
    id: str
    kind: str
    tenant_id: str
    state: Literal["queued", "running", "succeeded", "failed"]
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class JobListResponse(BaseModel):
    jobs: list[JobResponse]
    total: int


class ModelInfo(BaseModel):
    name: str
    backend: Literal["mock", "openclip", "siglip2"]
    dimension: int
    max_batch_size: int
    index_size: int
    vector_store: str = "sqlite"
    architecture: str
    device: str
    precision: str
    multilingual: bool
    max_text_tokens: int
    image_size: int | None
    cache_entries: int
    cache_capacity: int
    text_cache_hits: int
    image_cache_hits: int


class ModelPreset(BaseModel):
    id: str
    backend: Literal["openclip", "siglip2"]
    quality: Literal["balanced", "high", "maximum"]
    multilingual: bool
    description: str


class ModelCatalogResponse(BaseModel):
    presets: list[ModelPreset]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    model_ready: bool
