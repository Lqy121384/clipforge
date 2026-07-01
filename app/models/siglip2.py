from io import BytesIO
from typing import Any, cast

from PIL import Image

from app.models.base import ModelCapabilities


class Siglip2EmbeddingBackend:
    """Hugging Face SigLIP2 adapter for multilingual, modern retrieval models."""

    def __init__(
        self,
        model_id: str,
        device: str,
        precision: str,
        attention: str,
        compile_model: bool,
    ) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("SigLIP2 requires the 'ml' extra: pip install '.[ml]'") from exc

        self._torch = torch
        self._device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu"
            if device == "auto"
            else device
        )
        if precision == "auto":
            precision = "bf16" if self._device.startswith("cuda") else "fp32"
        self._precision = precision
        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[precision]
        kwargs: dict[str, Any] = {"dtype": dtype}
        if attention != "auto":
            kwargs["attn_implementation"] = attention
        self._processor = AutoProcessor.from_pretrained(model_id)  # type: ignore[no-untyped-call]
        self._model = AutoModel.from_pretrained(model_id, **kwargs).to(self._device).eval()
        config = self._model.config
        self.dimension = int(getattr(config, "projection_dim", config.vision_config.hidden_size))
        image_size = getattr(config.vision_config, "image_size", None)
        self.name = f"siglip2/{model_id}"
        self.capabilities = ModelCapabilities(
            architecture=getattr(config, "model_type", "siglip2"),
            device=self._device,
            precision=precision,
            multilingual=True,
            max_text_tokens=64,
            image_size=int(image_size) if image_size else None,
        )
        if compile_model and hasattr(torch, "compile"):
            self._model = torch.compile(self._model, mode="reduce-overhead")

    def _inputs_to_device(self, inputs: Any) -> dict[str, Any]:
        return {key: value.to(self._device) for key, value in inputs.items()}

    def _features(self, output: Any) -> Any:
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        return output

    def _normalize(self, features: Any) -> list[list[float]]:
        features = self._features(features)
        features = features / features.norm(p=2, dim=-1, keepdim=True)
        return cast(list[list[float]], features.detach().cpu().float().tolist())

    def encode_texts(self, texts: list[str]) -> list[list[float]]:
        inputs = self._processor(
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt",
        )
        inputs = self._inputs_to_device(inputs)
        with self._torch.inference_mode():
            return self._normalize(self._model.get_text_features(**inputs))

    def encode_images(self, images: list[bytes]) -> list[list[float]]:
        pil_images = [Image.open(BytesIO(payload)).convert("RGB") for payload in images]
        inputs = self._processor(images=pil_images, return_tensors="pt")
        inputs = self._inputs_to_device(inputs)
        with self._torch.inference_mode():
            return self._normalize(self._model.get_image_features(**inputs))

    def close(self) -> None:
        if self._device.startswith("cuda"):
            self._torch.cuda.empty_cache()
