import asyncio

from app.models.mock import MockEmbeddingBackend
from app.services.inference import InferenceEngine


def test_inference_cache_reuses_text_and_image_embeddings() -> None:
    engine = InferenceEngine(MockEmbeddingBackend(8), max_concurrency=1, cache_size=4)
    first_text = engine.text_sync(["same query"])
    second_text = engine.text_sync(["same query"])
    first_image = engine.image_sync([b"same image"])
    second_image = engine.image_sync([b"same image"])

    assert first_text == second_text
    assert first_image == second_image
    assert engine.stats.text_hits == 1
    assert engine.stats.image_hits == 1
    assert engine.stats.cache_entries == 2


def test_async_inference_contract() -> None:
    engine = InferenceEngine(MockEmbeddingBackend(8), max_concurrency=1, cache_size=0)
    result = asyncio.run(engine.text(["query"]))
    assert len(result) == 1
    assert len(result[0]) == 8
