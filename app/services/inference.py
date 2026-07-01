import asyncio
import hashlib
import threading
from collections import OrderedDict
from dataclasses import dataclass

from app.models.base import EmbeddingBackend


@dataclass(frozen=True, slots=True)
class InferenceStats:
    text_hits: int
    text_misses: int
    image_hits: int
    image_misses: int
    cache_entries: int
    cache_capacity: int


class InferenceEngine:
    """Concurrency-safe bridge between async HTTP and synchronous ML runtimes."""

    def __init__(
        self,
        model: EmbeddingBackend,
        max_concurrency: int,
        cache_size: int,
    ) -> None:
        self._model = model
        self._async_slots = asyncio.Semaphore(max_concurrency)
        self._model_lock = threading.RLock()
        self._cache_size = cache_size
        self._text_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._image_cache: OrderedDict[str, list[float]] = OrderedDict()
        self._text_hits = 0
        self._text_misses = 0
        self._image_hits = 0
        self._image_misses = 0

    def _get(
        self,
        cache: OrderedDict[str, list[float]],
        key: str,
    ) -> list[float] | None:
        value = cache.get(key)
        if value is not None:
            cache.move_to_end(key)
            return list(value)
        return None

    def _put(
        self,
        cache: OrderedDict[str, list[float]],
        key: str,
        value: list[float],
    ) -> None:
        if self._cache_size <= 0:
            return
        cache[key] = list(value)
        cache.move_to_end(key)
        while len(cache) > self._cache_size:
            cache.popitem(last=False)

    def text_sync(self, texts: list[str]) -> list[list[float]]:
        with self._model_lock:
            keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
            results: list[list[float] | None] = []
            misses: list[str] = []
            miss_positions: list[int] = []
            for index, (key, text) in enumerate(zip(keys, texts, strict=True)):
                cached = self._get(self._text_cache, key)
                results.append(cached)
                if cached is None:
                    self._text_misses += 1
                    misses.append(text)
                    miss_positions.append(index)
                else:
                    self._text_hits += 1
            if misses:
                encoded = self._model.encode_texts(misses)
                for position, vector in zip(miss_positions, encoded, strict=True):
                    results[position] = vector
                    self._put(self._text_cache, keys[position], vector)
            return [list(vector) for vector in results if vector is not None]

    def image_sync(self, images: list[bytes]) -> list[list[float]]:
        with self._model_lock:
            keys = [hashlib.sha256(image).hexdigest() for image in images]
            results: list[list[float] | None] = []
            misses: list[bytes] = []
            miss_positions: list[int] = []
            for index, (key, image) in enumerate(zip(keys, images, strict=True)):
                cached = self._get(self._image_cache, key)
                results.append(cached)
                if cached is None:
                    self._image_misses += 1
                    misses.append(image)
                    miss_positions.append(index)
                else:
                    self._image_hits += 1
            if misses:
                encoded = self._model.encode_images(misses)
                for position, vector in zip(miss_positions, encoded, strict=True):
                    results[position] = vector
                    self._put(self._image_cache, keys[position], vector)
            return [list(vector) for vector in results if vector is not None]

    async def text(self, texts: list[str]) -> list[list[float]]:
        async with self._async_slots:
            return await asyncio.to_thread(self.text_sync, texts)

    async def image(self, images: list[bytes]) -> list[list[float]]:
        async with self._async_slots:
            return await asyncio.to_thread(self.image_sync, images)

    @property
    def stats(self) -> InferenceStats:
        with self._model_lock:
            return InferenceStats(
                text_hits=self._text_hits,
                text_misses=self._text_misses,
                image_hits=self._image_hits,
                image_misses=self._image_misses,
                cache_entries=len(self._text_cache) + len(self._image_cache),
                cache_capacity=self._cache_size * 2,
            )
