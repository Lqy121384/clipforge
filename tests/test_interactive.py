import pytest

from app.services.interactive import estimate_uncertainty, refine_query


def test_uncertainty_detects_ambiguous_distribution() -> None:
    estimate = estimate_uncertainty(
        [0.51, 0.50, 0.49],
        temperature=0.07,
        margin_threshold=0.12,
        entropy_threshold=0.78,
    )
    assert estimate.needs_clarification is True
    assert estimate.normalized_entropy > 0.9
    assert estimate.reason == "low_margin_and_high_entropy"


def test_uncertainty_accepts_clear_winner() -> None:
    estimate = estimate_uncertainty(
        [0.9, 0.2, 0.1],
        temperature=0.07,
        margin_threshold=0.12,
        entropy_threshold=0.78,
    )
    assert estimate.needs_clarification is False
    assert estimate.confidence > 0.9


def test_relevance_feedback_moves_query_toward_positive_item() -> None:
    query = [1.0, 0.0]
    refined, drift = refine_query(
        query,
        positive_vectors=[(0.0, 1.0)],
        negative_vectors=[(1.0, 0.0)],
        alpha=1.0,
        beta=1.0,
        gamma=0.5,
    )
    assert refined[1] > refined[0]
    assert sum(value**2 for value in refined) ** 0.5 == pytest.approx(1.0)
    assert drift > 0
