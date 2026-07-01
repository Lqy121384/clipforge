from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    architecture: str
    device: str
    precision: str
    multilingual: bool
    max_text_tokens: int
    image_size: int | None


class EmbeddingBackend(Protocol):
    name: str
    dimension: int
    capabilities: ModelCapabilities

    def encode_texts(self, texts: list[str]) -> list[list[float]]: ...

    def encode_images(self, images: list[bytes]) -> list[list[float]]: ...

    def close(self) -> None: ...
