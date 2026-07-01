import heapq
import math
import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class IndexRecord:
    item_id: str
    vector: tuple[float, ...]
    metadata: dict[str, Any]


class InMemoryVectorIndex:
    def __init__(self, dimension: int, capacity: int) -> None:
        self.dimension = dimension
        self.capacity = capacity
        self._records: dict[str, IndexRecord] = {}
        self._lock = threading.RLock()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._records)

    def upsert(self, records: list[IndexRecord]) -> int:
        with self._lock:
            new_ids = {record.item_id for record in records} - self._records.keys()
            if len(self._records) + len(new_ids) > self.capacity:
                raise ValueError(f"Index capacity of {self.capacity} would be exceeded")
            for record in records:
                if len(record.vector) != self.dimension:
                    raise ValueError(
                        f"Vector dimension {len(record.vector)} does not match {self.dimension}"
                    )
                self._records[record.item_id] = record
            return len(records)

    def delete(self, item_ids: list[str]) -> int:
        with self._lock:
            removed = sum(self._records.pop(item_id, None) is not None for item_id in item_ids)
            return removed

    def search(
        self, vector: list[float], limit: int, metadata_filter: dict[str, Any] | None = None
    ) -> list[tuple[IndexRecord, float]]:
        if len(vector) != self.dimension:
            raise ValueError("Query vector has the wrong dimension")
        query_norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        with self._lock:
            records = list(self._records.values())

        scored: list[tuple[float, str, IndexRecord]] = []
        for record in records:
            if metadata_filter and not all(
                record.metadata.get(key) == value for key, value in metadata_filter.items()
            ):
                continue
            dot = sum(left * right for left, right in zip(vector, record.vector, strict=True))
            record_norm = math.sqrt(sum(value * value for value in record.vector)) or 1.0
            score = dot / (query_norm * record_norm)
            scored.append((score, record.item_id, record))
        best = heapq.nlargest(limit, scored, key=lambda item: (item[0], item[1]))
        return [(record, score) for score, _, record in best]
