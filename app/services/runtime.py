from dataclasses import dataclass
from io import BytesIO

from PIL import Image

from app.core.config import Settings
from app.models.base import EmbeddingBackend
from app.models.mock import MockEmbeddingBackend
from app.services.inference import InferenceEngine
from app.services.jobs import JobManager
from app.services.vector_store import SQLiteVectorStore


@dataclass(slots=True)
class Runtime:
    model: EmbeddingBackend
    inference: InferenceEngine
    store: SQLiteVectorStore
    jobs: JobManager


def create_runtime(settings: Settings) -> Runtime:
    if settings.model_backend == "openclip":
        from app.models.openclip import OpenClipEmbeddingBackend

        model: EmbeddingBackend = OpenClipEmbeddingBackend(
            settings.model_name,
            settings.model_pretrained,
            settings.model_device,
            settings.model_precision,
            settings.model_compile,
        )
    elif settings.model_backend == "siglip2":
        from app.models.siglip2 import Siglip2EmbeddingBackend

        model = Siglip2EmbeddingBackend(
            settings.model_id,
            settings.model_device,
            settings.model_precision,
            settings.model_attention,
            settings.model_compile,
        )
    else:
        model = MockEmbeddingBackend(settings.embedding_dimension)

    runtime = Runtime(
        model=model,
        inference=InferenceEngine(
            model,
            settings.max_concurrent_inference,
            settings.inference_cache_size,
        ),
        store=SQLiteVectorStore(
            settings.vector_store_path,
            model.dimension,
            settings.index_capacity,
        ),
        jobs=JobManager(settings.job_workers, settings.job_history_limit),
    )
    if settings.model_warmup:
        runtime.inference.text_sync(["a photo used to warm up the model"])
        image = Image.new("RGB", (224, 224), color=(127, 127, 127))
        buffer = BytesIO()
        image.save(buffer, format="JPEG")
        runtime.inference.image_sync([buffer.getvalue()])
    return runtime
