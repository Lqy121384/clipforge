from __future__ import annotations

# ruff: noqa: E402

import argparse
import gc
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import psutil
import torch
from PIL import Image
from torchvision.datasets import CIFAR10

from app.models.base import EmbeddingBackend
from app.models.openclip import OpenClipEmbeddingBackend
from app.models.siglip2 import Siglip2EmbeddingBackend

LABELS = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

PROMPT_TEMPLATES = [
    "a photo of a {}.",
    "a blurry photo of a {}.",
    "a close-up photo of a {}.",
    "a low resolution photo of a {}.",
    "an image of a {}.",
]


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    display_name: str
    backend: str
    model_id: str


@dataclass(slots=True)
class EvaluationResult:
    model: str
    backend: str
    model_id: str
    dataset: str
    sample_count: int
    seed: int
    batch_size: int
    prompt_templates: int
    top1_accuracy: float
    top5_accuracy: float
    macro_f1: float
    load_seconds: float
    encode_seconds: float
    latency_mean_ms: float
    latency_p50_ms: float
    latency_p95_ms: float
    throughput_images_per_second: float
    rss_memory_mb: float
    dimension: int
    device: str
    precision: str
    confusion_matrix: list[list[int]]


MODEL_SPECS = {
    "siglip2": ModelSpec(
        key="siglip2",
        display_name="SigLIP2 Base",
        backend="siglip2",
        model_id="google/siglip2-base-patch16-224",
    ),
    "openclip": ModelSpec(
        key="openclip",
        display_name="OpenCLIP ViT-B/32",
        backend="openclip",
        model_id="ViT-B-32/laion2b_s34b_b79k",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproducible CLIPForge zero-shot benchmark on CIFAR-10."
    )
    parser.add_argument(
        "--models",
        default="siglip2,openclip",
        help="Comma-separated model keys: siglip2,openclip",
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=Path("data/benchmarks"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evaluation"))
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--precision",
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
    )
    return parser.parse_args()


def stratified_indices(targets: list[int], sample_count: int, seed: int) -> list[int]:
    if sample_count < len(LABELS):
        raise ValueError(f"samples must be at least {len(LABELS)}")
    if sample_count > len(targets):
        raise ValueError(f"samples cannot exceed dataset size {len(targets)}")
    rng = random.Random(seed)
    groups: dict[int, list[int]] = {label: [] for label in range(len(LABELS))}
    for index, label in enumerate(targets):
        groups[label].append(index)
    for values in groups.values():
        rng.shuffle(values)
    base, remainder = divmod(sample_count, len(LABELS))
    selected: list[int] = []
    for label in range(len(LABELS)):
        count = base + (1 if label < remainder else 0)
        selected.extend(groups[label][:count])
    rng.shuffle(selected)
    return selected


def image_to_jpeg(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def build_label_vectors(model: EmbeddingBackend) -> list[list[float]]:
    prompts = [template.format(label) for label in LABELS for template in PROMPT_TEMPLATES]
    encoded = model.encode_texts(prompts)
    label_vectors: list[list[float]] = []
    width = len(PROMPT_TEMPLATES)
    for offset in range(0, len(encoded), width):
        group = encoded[offset : offset + width]
        averaged = [sum(values) / width for values in zip(*group, strict=True)]
        label_vectors.append(normalize(averaged))
    return label_vectors


def create_model(spec: ModelSpec, device: str, precision: str) -> EmbeddingBackend:
    if spec.backend == "siglip2":
        return Siglip2EmbeddingBackend(
            model_id=spec.model_id,
            device=device,
            precision=precision,
            attention="sdpa",
            compile_model=False,
        )
    model_name, pretrained = spec.model_id.split("/", 1)
    return OpenClipEmbeddingBackend(
        model_name=model_name,
        pretrained=pretrained,
        device=device,
        precision=precision,
        compile_model=False,
    )


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[position]


def macro_f1(confusion: list[list[int]]) -> float:
    scores: list[float] = []
    for label in range(len(LABELS)):
        true_positive = confusion[label][label]
        false_positive = sum(confusion[row][label] for row in range(len(LABELS))) - true_positive
        false_negative = sum(confusion[label]) - true_positive
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(2 * true_positive / denominator if denominator else 0.0)
    return statistics.fmean(scores)


def evaluate(
    spec: ModelSpec,
    dataset: CIFAR10,
    indices: list[int],
    batch_size: int,
    seed: int,
    device: str,
    precision: str,
) -> EvaluationResult:
    print(f"\n[{spec.display_name}] loading {spec.model_id}", flush=True)
    load_started = time.perf_counter()
    model = create_model(spec, device, precision)
    label_vectors = build_label_vectors(model)
    load_seconds = time.perf_counter() - load_started

    warmup_images = [image_to_jpeg(dataset[indices[0]][0])]
    model.encode_images(warmup_images)

    confusion = [[0 for _ in LABELS] for _ in LABELS]
    top1_correct = 0
    top5_correct = 0
    latencies_ms: list[float] = []
    encode_seconds = 0.0

    for offset in range(0, len(indices), batch_size):
        batch_indices = indices[offset : offset + batch_size]
        images = [image_to_jpeg(dataset[index][0]) for index in batch_indices]
        targets = [int(dataset[index][1]) for index in batch_indices]
        started = time.perf_counter()
        image_vectors = model.encode_images(images)
        elapsed = time.perf_counter() - started
        encode_seconds += elapsed
        latencies_ms.extend([elapsed * 1000 / len(images)] * len(images))

        for vector, target in zip(image_vectors, targets, strict=True):
            scores = [
                sum(left * right for left, right in zip(vector, label_vector, strict=True))
                for label_vector in label_vectors
            ]
            ranked = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
            prediction = ranked[0]
            confusion[target][prediction] += 1
            top1_correct += int(prediction == target)
            top5_correct += int(target in ranked[:5])

        completed = min(offset + len(batch_indices), len(indices))
        print(
            f"\r[{spec.display_name}] {completed}/{len(indices)} ({completed / len(indices):.0%})",
            end="",
            flush=True,
        )
    print()

    process = psutil.Process(os.getpid())
    result = EvaluationResult(
        model=spec.display_name,
        backend=spec.backend,
        model_id=spec.model_id,
        dataset="CIFAR-10 test stratified subset",
        sample_count=len(indices),
        seed=seed,
        batch_size=batch_size,
        prompt_templates=len(PROMPT_TEMPLATES),
        top1_accuracy=top1_correct / len(indices),
        top5_accuracy=top5_correct / len(indices),
        macro_f1=macro_f1(confusion),
        load_seconds=load_seconds,
        encode_seconds=encode_seconds,
        latency_mean_ms=statistics.fmean(latencies_ms),
        latency_p50_ms=statistics.median(latencies_ms),
        latency_p95_ms=percentile(latencies_ms, 0.95),
        throughput_images_per_second=len(indices) / encode_seconds,
        rss_memory_mb=process.memory_info().rss / (1024 * 1024),
        dimension=model.dimension,
        device=model.capabilities.device,
        precision=model.capabilities.precision,
        confusion_matrix=confusion,
    )
    model.close()
    del model
    gc.collect()
    return result


def environment_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "requested_device": args.device,
        "requested_precision": args.precision,
        "sample_protocol": "deterministic stratified sampling, equal class allocation",
        "latency_protocol": "model image encoding only; preprocessing and model loading excluded",
    }


def plot_results(results: list[EvaluationResult], output_dir: Path) -> None:
    names = [result.model for result in results]
    colors = ["#9BC53D", "#2D7DD2", "#E84855", "#F9A03F"][: len(results)]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    accuracy_metrics = {
        "Top-1 Accuracy": [result.top1_accuracy * 100 for result in results],
        "Top-5 Accuracy": [result.top5_accuracy * 100 for result in results],
        "Macro-F1": [result.macro_f1 * 100 for result in results],
    }
    x_positions = list(range(len(results)))
    width = 0.23
    for metric_index, (label, values) in enumerate(accuracy_metrics.items()):
        positions = [position + (metric_index - 1) * width for position in x_positions]
        axes[0].bar(positions, values, width, label=label, alpha=0.9)
    axes[0].set_xticks(x_positions, names)
    axes[0].set_ylabel("Score (%)")
    axes[0].set_title("Zero-shot quality · CIFAR-10")
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    throughput = [result.throughput_images_per_second for result in results]
    bars = axes[1].bar(names, throughput, color=colors)
    axes[1].set_ylabel("Images / second")
    axes[1].set_title("CPU/GPU encoding throughput")
    axes[1].grid(axis="y", alpha=0.2)
    for bar, value in zip(bars, throughput, strict=True):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    figure.suptitle(
        f"CLIPForge model benchmark · {results[0].sample_count} stratified samples",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    figure.savefig(output_dir / "model_comparison.png", dpi=180, bbox_inches="tight")
    figure.savefig(output_dir / "model_comparison.svg", bbox_inches="tight")
    plt.close(figure)

    latency_figure, latency_axis = plt.subplots(figsize=(7.2, 4.8))
    x = list(range(len(results)))
    width = 0.34
    latency_axis.bar(
        [value - width / 2 for value in x],
        [result.latency_mean_ms for result in results],
        width,
        label="Mean",
        color="#2D7DD2",
    )
    latency_axis.bar(
        [value + width / 2 for value in x],
        [result.latency_p95_ms for result in results],
        width,
        label="P95",
        color="#E84855",
    )
    latency_axis.set_xticks(x, names)
    latency_axis.set_ylabel("Milliseconds / image")
    latency_axis.set_title("Image encoding latency")
    latency_axis.grid(axis="y", alpha=0.2)
    latency_axis.legend(frameon=False)
    latency_figure.tight_layout()
    latency_figure.savefig(output_dir / "latency_comparison.png", dpi=180, bbox_inches="tight")
    latency_figure.savefig(output_dir / "latency_comparison.svg", bbox_inches="tight")
    plt.close(latency_figure)


def write_report(
    results: list[EvaluationResult],
    environment: dict[str, Any],
    output_dir: Path,
) -> None:
    payload = {
        "schema_version": "1.0",
        "environment": environment,
        "results": [asdict(result) for result in results],
    }
    (output_dir / "benchmark_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rows = "\n".join(
        (
            f"| {result.model} | {result.top1_accuracy:.2%} | "
            f"{result.top5_accuracy:.2%} | {result.macro_f1:.2%} | "
            f"{result.latency_mean_ms:.1f} | {result.latency_p95_ms:.1f} | "
            f"{result.throughput_images_per_second:.2f} | "
            f"{result.rss_memory_mb:.0f} |"
        )
        for result in results
    )
    report = f"""# CLIPForge Model Benchmark

This report is generated from real model inference. No metric is manually entered.

## Protocol

- Dataset: CIFAR-10 public test split
- Subset: {results[0].sample_count} deterministic stratified samples
- Seed: {results[0].seed}
- Prompt ensemble: {results[0].prompt_templates} templates per class
- Device: {results[0].device}
- Latency: model image encoding only; warm-up, preprocessing and loading excluded
- Top-1/Top-5: cosine similarity against averaged, normalized class prompt embeddings

## Results

| Model | Top-1 | Top-5 | Macro-F1 | Mean ms/image | P95 ms/image | Images/s | RSS MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
{rows}

![Model comparison](model_comparison.png)

![Latency comparison](latency_comparison.png)

## Reproduce

```powershell
python scripts/evaluate_models.py --models siglip2,openclip \\
  --samples {results[0].sample_count} --batch-size {results[0].batch_size} \\
  --seed {results[0].seed}
```

Raw machine-readable results are stored in `benchmark_results.json`.
"""
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.samples <= 0 or args.batch_size <= 0:
        raise ValueError("samples and batch-size must be positive")
    selected_keys = [value.strip() for value in args.models.split(",") if value.strip()]
    unknown = set(selected_keys) - MODEL_SPECS.keys()
    if unknown:
        raise ValueError(f"Unknown model keys: {sorted(unknown)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    print("Downloading/loading CIFAR-10 public test split...", flush=True)
    dataset = CIFAR10(root=args.data_dir, train=False, download=True)
    indices = stratified_indices(dataset.targets, args.samples, args.seed)
    results = [
        evaluate(
            MODEL_SPECS[key],
            dataset,
            indices,
            args.batch_size,
            args.seed,
            args.device,
            args.precision,
        )
        for key in selected_keys
    ]
    environment = environment_payload(args)
    plot_results(results, args.output_dir)
    write_report(results, environment, args.output_dir)
    print(f"\nBenchmark complete: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
