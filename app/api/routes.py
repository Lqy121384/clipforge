import base64
import binascii
import math
import re
import time
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status

from app.api.dependencies import get_runtime, get_tenant_id
from app.api.schemas import (
    BatchIndexJobRequest,
    ClarificationOption,
    ClarificationPrompt,
    ClassificationLabel,
    ClassificationRequest,
    ClassificationResponse,
    ClassificationResult,
    CollectionCreateRequest,
    CollectionListResponse,
    CollectionResponse,
    EmbeddingResponse,
    HealthResponse,
    ImageEmbeddingRequest,
    IndexDeleteRequest,
    IndexImageRequest,
    IndexMutationResponse,
    IndexUpsertRequest,
    InteractiveSearchRequest,
    InteractiveSearchResponse,
    JobListResponse,
    JobResponse,
    ModelCatalogResponse,
    ModelInfo,
    ModelPreset,
    MultimodalSearchRequest,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SimilarityRequest,
    SimilarityResponse,
    TextEmbeddingRequest,
    UncertaintyResponse,
)
from app.core.config import Settings, get_settings
from app.core.security import require_api_key
from app.services.index import IndexRecord
from app.services.interactive import estimate_uncertainty, refine_query
from app.services.runtime import Runtime

router = APIRouter()
protected = APIRouter(dependencies=[Depends(require_api_key)])


def _decode_image(value: str, max_bytes: int) -> bytes:
    encoded = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={"code": "invalid_image", "message": "Image must be valid base64."},
        ) from exc
    if not payload or len(payload) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail={
                "code": "image_size_invalid",
                "message": f"Each image must contain 1 to {max_bytes} bytes.",
            },
        )
    return payload


def _validate_batch(count: int, settings: Settings) -> None:
    if count > settings.max_batch_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "batch_too_large",
                "message": f"Maximum batch size is {settings.max_batch_size}.",
            },
        )


def _validate_collection(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", name):
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_collection", "message": "Invalid collection name."},
        )
    return name


def _collection_response(info: object) -> CollectionResponse:
    return CollectionResponse.model_validate(info, from_attributes=True)


def _job_response(job: object) -> JobResponse:
    return JobResponse.model_validate(job, from_attributes=True)


@router.get("/health/live", response_model=HealthResponse, tags=["Operations"])
async def live(request: Request) -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=request.app.version,
        model_ready=hasattr(request.app.state, "runtime"),
    )


@router.get("/health/ready", response_model=HealthResponse, tags=["Operations"])
async def ready(request: Request) -> HealthResponse:
    is_ready = hasattr(request.app.state, "runtime")
    return HealthResponse(
        status="ok" if is_ready else "degraded",
        version=request.app.version,
        model_ready=is_ready,
    )


@protected.get("/model", response_model=ModelInfo, tags=["Model"])
async def model_info(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    settings: Annotated[Settings, Depends(get_settings)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> ModelInfo:
    inference_stats = runtime.inference.stats
    return ModelInfo(
        name=runtime.model.name,
        backend=settings.model_backend,
        dimension=runtime.model.dimension,
        max_batch_size=settings.max_batch_size,
        index_size=runtime.store.count(tenant_id),
        architecture=runtime.model.capabilities.architecture,
        device=runtime.model.capabilities.device,
        precision=runtime.model.capabilities.precision,
        multilingual=runtime.model.capabilities.multilingual,
        max_text_tokens=runtime.model.capabilities.max_text_tokens,
        image_size=runtime.model.capabilities.image_size,
        cache_entries=inference_stats.cache_entries,
        cache_capacity=inference_stats.cache_capacity,
        text_cache_hits=inference_stats.text_hits,
        image_cache_hits=inference_stats.image_hits,
    )


@protected.get("/models/catalog", response_model=ModelCatalogResponse, tags=["Model"])
async def model_catalog() -> ModelCatalogResponse:
    return ModelCatalogResponse(
        presets=[
            ModelPreset(
                id="google/siglip2-base-patch16-224",
                backend="siglip2",
                quality="balanced",
                multilingual=True,
                description="Modern multilingual retrieval with a practical memory footprint.",
            ),
            ModelPreset(
                id="google/siglip2-so400m-patch14-384",
                backend="siglip2",
                quality="maximum",
                multilingual=True,
                description="Higher-resolution 400M-parameter SigLIP2 for quality-first search.",
            ),
            ModelPreset(
                id="EVA02-L-14/merged2b_s4b_b131k",
                backend="openclip",
                quality="high",
                multilingual=False,
                description="Strong OpenCLIP EVA checkpoint for English-heavy workloads.",
            ),
        ]
    )


@protected.post("/embeddings/text", response_model=EmbeddingResponse, tags=["Embeddings"])
async def embed_text(
    body: TextEmbeddingRequest,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingResponse:
    _validate_batch(len(body.texts), settings)
    started = time.perf_counter()
    embeddings = await runtime.inference.text(body.texts)
    return EmbeddingResponse(
        model=runtime.model.name,
        dimension=runtime.model.dimension,
        embeddings=embeddings,
        count=len(embeddings),
        duration_ms=(time.perf_counter() - started) * 1000,
    )


@protected.post("/embeddings/image", response_model=EmbeddingResponse, tags=["Embeddings"])
async def embed_image(
    body: ImageEmbeddingRequest,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> EmbeddingResponse:
    _validate_batch(len(body.images), settings)
    payloads = [_decode_image(value, settings.max_image_bytes) for value in body.images]
    started = time.perf_counter()
    embeddings = await runtime.inference.image(payloads)
    return EmbeddingResponse(
        model=runtime.model.name,
        dimension=runtime.model.dimension,
        embeddings=embeddings,
        count=len(embeddings),
        duration_ms=(time.perf_counter() - started) * 1000,
    )


@protected.post("/similarity", response_model=SimilarityResponse, tags=["Embeddings"])
async def similarity(
    body: SimilarityRequest,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> SimilarityResponse:
    _validate_batch(len(body.texts), settings)
    _validate_batch(len(body.images), settings)
    images = [_decode_image(value, settings.max_image_bytes) for value in body.images]
    started = time.perf_counter()
    text_vectors = await runtime.inference.text(body.texts)
    image_vectors = await runtime.inference.image(images)
    scores = [
        [
            sum(left * right for left, right in zip(text, image, strict=True))
            / (
                (math.sqrt(sum(value * value for value in text)) or 1.0)
                * (math.sqrt(sum(value * value for value in image)) or 1.0)
            )
            for image in image_vectors
        ]
        for text in text_vectors
    ]
    return SimilarityResponse(
        model=runtime.model.name,
        scores=scores,
        text_count=len(text_vectors),
        image_count=len(image_vectors),
        duration_ms=(time.perf_counter() - started) * 1000,
    )


@protected.post(
    "/classifications/zero-shot",
    response_model=ClassificationResponse,
    tags=["Embeddings"],
)
async def zero_shot_classification(
    body: ClassificationRequest,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ClassificationResponse:
    _validate_batch(len(body.images), settings)
    prompt_count = len(body.labels) * len(body.templates)
    if prompt_count > 4096:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "prompt_ensemble_too_large",
                "message": "labels × templates must not exceed 4096.",
            },
        )
    images = [_decode_image(value, settings.max_image_bytes) for value in body.images]
    prompts = [template.format(label) for label in body.labels for template in body.templates]
    started = time.perf_counter()
    prompt_vectors = await runtime.inference.text(prompts)
    label_vectors: list[list[float]] = []
    template_count = len(body.templates)
    for offset in range(0, len(prompt_vectors), template_count):
        group = prompt_vectors[offset : offset + template_count]
        averaged = [sum(values) / template_count for values in zip(*group, strict=True)]
        norm = math.sqrt(sum(value * value for value in averaged)) or 1.0
        label_vectors.append([value / norm for value in averaged])
    image_vectors = await runtime.inference.image(images)
    results: list[ClassificationResult] = []
    for image_index, image_vector in enumerate(image_vectors):
        scores = [
            sum(left * right for left, right in zip(image_vector, label, strict=True))
            for label in label_vectors
        ]
        shifted = [score / body.temperature for score in scores]
        peak = max(shifted)
        exponentials = [math.exp(value - peak) for value in shifted]
        denominator = sum(exponentials)
        ranked = sorted(
            zip(body.labels, scores, exponentials, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )[: min(body.top_k, len(body.labels))]
        results.append(
            ClassificationResult(
                image_index=image_index,
                predictions=[
                    ClassificationLabel(
                        label=label,
                        score=score,
                        probability=exponential / denominator,
                    )
                    for label, score, exponential in ranked
                ],
            )
        )
    return ClassificationResponse(
        model=runtime.model.name,
        prompt_count=prompt_count,
        results=results,
        duration_ms=(time.perf_counter() - started) * 1000,
    )


@protected.post("/index/text", response_model=IndexMutationResponse, tags=["Search"])
async def upsert_text(
    body: IndexUpsertRequest,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    settings: Annotated[Settings, Depends(get_settings)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> IndexMutationResponse:
    _validate_batch(len(body.items), settings)
    vectors = await runtime.inference.text([item.text for item in body.items])
    records = [
        IndexRecord(
            item_id=item.id,
            vector=tuple(vector),
            metadata={**item.metadata, "_text": item.text},
        )
        for item, vector in zip(body.items, vectors, strict=True)
    ]
    try:
        affected = runtime.store.upsert(
            tenant_id,
            settings.default_collection,
            records,
            "text",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "index_error", "message": str(exc)},
        ) from exc
    size = runtime.store.get_collection(tenant_id, settings.default_collection).size
    return IndexMutationResponse(
        affected=affected,
        index_size=size,
        collection=settings.default_collection,
    )


@protected.post("/index/delete", response_model=IndexMutationResponse, tags=["Search"])
async def delete_items(
    body: IndexDeleteRequest,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    settings: Annotated[Settings, Depends(get_settings)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> IndexMutationResponse:
    affected = runtime.store.delete(tenant_id, settings.default_collection, body.ids)
    try:
        size = runtime.store.get_collection(tenant_id, settings.default_collection).size
    except KeyError:
        size = 0
    return IndexMutationResponse(
        affected=affected,
        index_size=size,
        collection=settings.default_collection,
    )


@protected.post("/search/text", response_model=SearchResponse, tags=["Search"])
async def search_text(
    body: SearchRequest,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    settings: Annotated[Settings, Depends(get_settings)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> SearchResponse:
    started = time.perf_counter()
    query_vector = (await runtime.inference.text([body.query]))[0]
    results = runtime.store.search(
        tenant_id,
        settings.default_collection,
        query_vector,
        body.limit,
        body.metadata_filter,
    )
    return SearchResponse(
        query=body.query,
        hits=[
            SearchHit(
                id=record.item_id,
                score=score,
                modality=modality,
                metadata=record.metadata,
            )
            for record, score, modality in results
        ],
        took_ms=(time.perf_counter() - started) * 1000,
        collection=settings.default_collection,
    )


@protected.post(
    "/collections",
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
    tags=["Collections"],
)
async def create_collection(
    body: CollectionCreateRequest,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> CollectionResponse:
    return _collection_response(runtime.store.create_collection(tenant_id, body.name))


@protected.get(
    "/collections",
    response_model=CollectionListResponse,
    tags=["Collections"],
)
async def list_collections(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> CollectionListResponse:
    collections = [_collection_response(info) for info in runtime.store.list_collections(tenant_id)]
    return CollectionListResponse(collections=collections, total=len(collections))


@protected.get(
    "/collections/{collection}",
    response_model=CollectionResponse,
    tags=["Collections"],
)
async def get_collection(
    collection: str,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> CollectionResponse:
    _validate_collection(collection)
    try:
        return _collection_response(runtime.store.get_collection(tenant_id, collection))
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "collection_not_found", "message": "Collection not found."},
        ) from exc


@protected.delete(
    "/collections/{collection}",
    status_code=status.HTTP_204_NO_CONTENT,
    tags=["Collections"],
)
async def delete_collection(
    collection: str,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> None:
    _validate_collection(collection)
    if not runtime.store.delete_collection(tenant_id, collection):
        raise HTTPException(
            status_code=404,
            detail={"code": "collection_not_found", "message": "Collection not found."},
        )


@protected.post(
    "/collections/{collection}/items/text",
    response_model=IndexMutationResponse,
    tags=["Collections"],
)
async def collection_upsert_text(
    collection: str,
    body: IndexUpsertRequest,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    settings: Annotated[Settings, Depends(get_settings)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> IndexMutationResponse:
    _validate_collection(collection)
    _validate_batch(len(body.items), settings)
    vectors = await runtime.inference.text([item.text for item in body.items])
    records = [
        IndexRecord(
            item_id=item.id,
            vector=tuple(vector),
            metadata={**item.metadata, "_text": item.text},
        )
        for item, vector in zip(body.items, vectors, strict=True)
    ]
    affected = runtime.store.upsert(tenant_id, collection, records, "text")
    size = runtime.store.get_collection(tenant_id, collection).size
    return IndexMutationResponse(affected=affected, index_size=size, collection=collection)


@protected.post(
    "/collections/{collection}/items/image",
    response_model=IndexMutationResponse,
    tags=["Collections"],
)
async def collection_upsert_image(
    collection: str,
    body: IndexImageRequest,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    settings: Annotated[Settings, Depends(get_settings)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> IndexMutationResponse:
    _validate_collection(collection)
    _validate_batch(len(body.items), settings)
    payloads = [_decode_image(item.image, settings.max_image_bytes) for item in body.items]
    vectors = await runtime.inference.image(payloads)
    records = [
        IndexRecord(item_id=item.id, vector=tuple(vector), metadata=item.metadata)
        for item, vector in zip(body.items, vectors, strict=True)
    ]
    affected = runtime.store.upsert(tenant_id, collection, records, "image")
    size = runtime.store.get_collection(tenant_id, collection).size
    return IndexMutationResponse(affected=affected, index_size=size, collection=collection)


@protected.post(
    "/collections/{collection}/search",
    response_model=SearchResponse,
    tags=["Collections"],
)
async def multimodal_search(
    collection: str,
    body: MultimodalSearchRequest,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    settings: Annotated[Settings, Depends(get_settings)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> SearchResponse:
    _validate_collection(collection)
    started = time.perf_counter()
    if body.query_type == "text":
        vector = (await runtime.inference.text([body.query]))[0]
        query_label = body.query
    else:
        payload = _decode_image(body.query, settings.max_image_bytes)
        vector = (await runtime.inference.image([payload]))[0]
        query_label = "base64:image"
    try:
        results = runtime.store.search(
            tenant_id,
            collection,
            vector,
            body.limit,
            body.metadata_filter,
            body.target_modality,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "collection_not_found", "message": "Collection not found."},
        ) from exc
    return SearchResponse(
        query=query_label,
        hits=[
            SearchHit(
                id=record.item_id,
                score=score,
                modality=modality,
                metadata=record.metadata,
            )
            for record, score, modality in results
        ],
        took_ms=(time.perf_counter() - started) * 1000,
        collection=collection,
    )


@protected.post(
    "/collections/{collection}/search/interactive",
    response_model=InteractiveSearchResponse,
    tags=["Human-in-the-loop Search"],
)
async def interactive_search(
    collection: str,
    body: InteractiveSearchRequest,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    settings: Annotated[Settings, Depends(get_settings)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> InteractiveSearchResponse:
    _validate_collection(collection)
    started = time.perf_counter()
    if body.query_type == "text":
        original_vector = (await runtime.inference.text([body.query]))[0]
        query_label = body.query
    else:
        payload = _decode_image(body.query, settings.max_image_bytes)
        original_vector = (await runtime.inference.image([payload]))[0]
        query_label = "base64:image"

    feedback_ids = body.feedback.positive_ids + body.feedback.negative_ids
    feedback_records = runtime.store.get_records(
        tenant_id,
        collection,
        feedback_ids,
    )
    found_ids = {record.item_id for record, _ in feedback_records}
    missing_ids = set(feedback_ids) - found_ids
    if missing_ids:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "feedback_items_not_found",
                "message": f"Feedback items not found: {sorted(missing_ids)}",
            },
        )
    vectors_by_id = {record.item_id: record.vector for record, _ in feedback_records}
    positive_vectors = [vectors_by_id[item_id] for item_id in body.feedback.positive_ids]
    negative_vectors = [vectors_by_id[item_id] for item_id in body.feedback.negative_ids]
    query_vector, query_drift = refine_query(
        original_vector,
        positive_vectors,
        negative_vectors,
        body.feedback.alpha,
        body.feedback.beta,
        body.feedback.gamma,
    )
    results = runtime.store.search(
        tenant_id,
        collection,
        query_vector,
        body.limit,
        body.metadata_filter,
        body.target_modality,
    )
    hits = [
        SearchHit(
            id=record.item_id,
            score=score,
            modality=modality,
            metadata=record.metadata,
        )
        for record, score, modality in results
    ]
    uncertainty = estimate_uncertainty(
        [hit.score for hit in hits],
        body.uncertainty_temperature,
        body.margin_threshold,
        body.entropy_threshold,
    )
    clarification = None
    if uncertainty.needs_clarification and len(hits) >= 2:
        options = []
        for hit in hits[:2]:
            label = (
                hit.metadata.get("_text")
                or hit.metadata.get("title")
                or hit.metadata.get("name")
                or hit.id
            )
            options.append(
                ClarificationOption(
                    id=hit.id,
                    label=str(label),
                    modality=hit.modality,
                    score=hit.score,
                )
            )
        clarification = ClarificationPrompt(
            question="Which result is closer to your intent?",
            options=options,
        )
    return InteractiveSearchResponse(
        query=query_label,
        hits=hits,
        took_ms=(time.perf_counter() - started) * 1000,
        collection=collection,
        uncertainty=UncertaintyResponse.model_validate(
            uncertainty,
            from_attributes=True,
        ),
        clarification=clarification,
        feedback_applied=bool(positive_vectors or negative_vectors),
        query_drift=query_drift,
    )


@protected.post(
    "/jobs/index/text",
    response_model=JobResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["Jobs"],
)
async def submit_text_index_job(
    body: BatchIndexJobRequest,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    settings: Annotated[Settings, Depends(get_settings)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> JobResponse:
    items = list(body.items)

    def operation() -> dict[str, object]:
        affected = 0
        for offset in range(0, len(items), settings.max_batch_size):
            batch = items[offset : offset + settings.max_batch_size]
            vectors = runtime.inference.text_sync([item.text for item in batch])
            records = [
                IndexRecord(
                    item_id=item.id,
                    vector=tuple(vector),
                    metadata={**item.metadata, "_text": item.text},
                )
                for item, vector in zip(batch, vectors, strict=True)
            ]
            affected += runtime.store.upsert(
                tenant_id,
                body.collection,
                records,
                "text",
            )
        size = runtime.store.get_collection(tenant_id, body.collection).size
        return {"affected": affected, "index_size": size, "collection": body.collection}

    job = runtime.jobs.submit("index.text", tenant_id, operation)
    return _job_response(job)


@protected.get("/jobs", response_model=JobListResponse, tags=["Jobs"])
async def list_jobs(
    runtime: Annotated[Runtime, Depends(get_runtime)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> JobListResponse:
    jobs = [_job_response(job) for job in runtime.jobs.list(tenant_id)]
    return JobListResponse(jobs=jobs, total=len(jobs))


@protected.get("/jobs/{job_id}", response_model=JobResponse, tags=["Jobs"])
async def get_job(
    job_id: str,
    runtime: Annotated[Runtime, Depends(get_runtime)],
    tenant_id: Annotated[str, Depends(get_tenant_id)],
) -> JobResponse:
    try:
        return _job_response(runtime.jobs.get(job_id, tenant_id))
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "job_not_found", "message": "Job not found."},
        ) from exc


router.include_router(protected)
