import hashlib
import math

from app.models.base import ModelCapabilities


class MockEmbeddingBackend:
    """Deterministic backend for local development and contract testing."""

    name = "deterministic-mock"

    def __init__(self, dimension: int) -> None:
        self.dimension = dimension
        self.capabilities = ModelCapabilities(
            architecture="mock",
            device="cpu",
            precision="fp32",
            multilingual=True,
            max_text_tokens=0,
            image_size=None,
        )

    def _embed(self, payload: bytes) -> list[float]:
        values: list[float] = []
        counter = 0
        while len(values) < self.dimension:
            digest = hashlib.sha256(payload + counter.to_bytes(4, "big")).digest()
            values.extend((byte / 127.5) - 1.0 for byte in digest)
            counter += 1
        values = values[: self.dimension]
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return [value / norm for value in values]

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(f"text:{text.strip()}".encode()) for text in texts]

    def encode_images(self, images: list[bytes]) -> list[list[float]]:
        return [self._embed(b"image:" + image) for image in images]

    def close(self) -> None:
        return None
