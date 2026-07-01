import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UncertaintyEstimate:
    margin: float
    normalized_entropy: float
    confidence: float
    needs_clarification: bool
    reason: str


def estimate_uncertainty(
    scores: list[float],
    temperature: float,
    margin_threshold: float,
    entropy_threshold: float,
) -> UncertaintyEstimate:
    """Estimate retrieval uncertainty from the returned score distribution."""
    if len(scores) < 2:
        return UncertaintyEstimate(
            margin=0.0,
            normalized_entropy=1.0,
            confidence=0.0,
            needs_clarification=True,
            reason="insufficient_results",
        )

    scaled = [score / temperature for score in scores]
    peak = max(scaled)
    exponentials = [math.exp(value - peak) for value in scaled]
    total = sum(exponentials) or 1.0
    probabilities = sorted(
        (value / total for value in exponentials),
        reverse=True,
    )
    margin = probabilities[0] - probabilities[1]
    entropy = -sum(
        probability * math.log(probability) for probability in probabilities if probability > 0
    )
    normalized_entropy = entropy / math.log(len(probabilities))
    margin_confidence = min(1.0, margin / margin_threshold)
    entropy_confidence = max(
        0.0,
        min(1.0, (1.0 - normalized_entropy) / (1.0 - entropy_threshold)),
    )
    confidence = (margin_confidence + entropy_confidence) / 2
    low_margin = margin < margin_threshold
    high_entropy = normalized_entropy > entropy_threshold
    if low_margin and high_entropy:
        reason = "low_margin_and_high_entropy"
    elif low_margin:
        reason = "low_margin"
    elif high_entropy:
        reason = "high_entropy"
    else:
        reason = "confident"
    return UncertaintyEstimate(
        margin=margin,
        normalized_entropy=normalized_entropy,
        confidence=confidence,
        needs_clarification=low_margin or high_entropy,
        reason=reason,
    )


def refine_query(
    query: list[float],
    positive_vectors: list[tuple[float, ...]],
    negative_vectors: list[tuple[float, ...]],
    alpha: float,
    beta: float,
    gamma: float,
) -> tuple[list[float], float]:
    """Apply Rocchio-style relevance feedback and return vector plus cosine drift."""
    if not positive_vectors and not negative_vectors:
        return list(query), 0.0

    dimension = len(query)
    refined = [alpha * value for value in query]
    if positive_vectors:
        for index in range(dimension):
            refined[index] += beta * (
                sum(vector[index] for vector in positive_vectors) / len(positive_vectors)
            )
    if negative_vectors:
        for index in range(dimension):
            refined[index] -= gamma * (
                sum(vector[index] for vector in negative_vectors) / len(negative_vectors)
            )
    norm = math.sqrt(sum(value * value for value in refined)) or 1.0
    refined = [value / norm for value in refined]
    query_norm = math.sqrt(sum(value * value for value in query)) or 1.0
    cosine = sum(left * right for left, right in zip(query, refined, strict=True))
    drift = 1.0 - cosine / query_norm
    return refined, max(0.0, min(2.0, drift))
