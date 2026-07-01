import pytest

from app.services.index import IndexRecord, InMemoryVectorIndex


def test_index_orders_cosine_similarity() -> None:
    index = InMemoryVectorIndex(dimension=2, capacity=3)
    index.upsert(
        [
            IndexRecord("near", (1.0, 0.0), {}),
            IndexRecord("far", (0.0, 1.0), {}),
        ]
    )
    results = index.search([0.9, 0.1], limit=2)
    assert [record.item_id for record, _ in results] == ["near", "far"]


def test_index_enforces_capacity() -> None:
    index = InMemoryVectorIndex(dimension=2, capacity=1)
    index.upsert([IndexRecord("one", (1.0, 0.0), {})])
    with pytest.raises(ValueError, match="capacity"):
        index.upsert([IndexRecord("two", (0.0, 1.0), {})])
