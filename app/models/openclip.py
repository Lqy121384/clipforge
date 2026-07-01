from io import BytesIO
from typing import Any, cast

from PIL import Image

from app.models.base import ModelCapabilities


class OpenClipEmbeddingBackend:
    name = "openclip"

    def __init__(
        self,
        model_name: str,
        pretrained: str | None,
        device: str,
        precision: str,
        compile_model: bool,
    ) -> None:
        try:
            import open_clip
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "OpenCLIP backend requires the 'ml' extra: pip install '.[ml]'"
            ) from exc

        self._torch = torch
        self._device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu"
            if device == "auto"
            else device
        )
        if precision == "auto":
            precision = "fp16" if self._device.startswith("cuda") else "fp32"
        self._precision = precision
        openclip_precision = {"fp32": "fp32", "fp16": "fp16", "bf16": "bf16"}[precision]
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self._device,
            precision=openclip_precision,
        )
        self._tokenizer = open_clip.get_tokenizer(model_name)
        self._model.eval()
        self.dimension = int(self._model.text_projection.shape[-1])
        self.name = f"openclip/{model_name}/{pretrained or 'random'}"
        image_size = getattr(self._model.visual, "image_size", None)
        if isinstance(image_size, tuple):
            image_size = image_size[0]
        self.capabilities = ModelCapabilities(
            architecture=model_name,
            device=self._device,
            precision=precision,
            multilingual="xlm" in model_name.lower() or "mt5" in model_name.lower(),
            max_text_tokens=int(getattr(self._model, "context_length", 77)),
            image_size=int(image_size) if image_size else None,
        )
        if compile_model and hasattr(torch, "compile"):
            self._model = torch.compile(self._model, mode="reduce-overhead")

    def _to_list(self, features: Any) -> list[list[float]]:
        features = features / features.norm(dim=-1, keepdim=True)
        return cast(list[list[float]], features.detach().cpu().float().tolist())

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        tokens = self._tokenizer(texts).to(self._device)
        with self._torch.inference_mode():
            return self._to_list(self._model.encode_text(tokens))

    def encode_images(self, images: list[bytes]) -> list[list[float]]:
        tensors = [
            self._preprocess(Image.open(BytesIO(payload)).convert("RGB")) for payload in images
        ]
        batch = self._torch.stack(tensors).to(self._device)
        with self._torch.inference_mode():
            return self._to_list(self._model.encode_image(batch))

    def close(self) -> None:
        if self._device.startswith("cuda"):
            self._torch.cuda.empty_cache()
