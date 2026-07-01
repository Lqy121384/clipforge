import hashlib
import heapq
import json
import math
import sqlite3
import threading
from array import array
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from app.services.index import IndexRecord


@dataclass(frozen=True, slots=True)
class CollectionInfo:
    name: str
    tenant_id: str
    dimension: int
    size: int
    created_at: str


class SQLiteVectorStore:
    """Persistent, tenant-isolated vector store with an intentionally small surface.

    SQLite is the zero-infrastructure provider. The service boundary makes it possible
    to replace this implementation with pgvector/Qdrant without changing the HTTP API.
    """

    def __init__(self, path: str, dimension: int, capacity: int) -> None:
        self.dimension = dimension
        self.capacity = capacity
        self._lock = threading.RLock()
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._migrate()

    def _migrate(self) -> None:
        with self._db:
            self._db.executescript(
                """
                CREATE TABLE IF NOT EXISTS collections (
                    tenant_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    dimension INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, name)
                );
                CREATE TABLE IF NOT EXISTS vectors (
                    tenant_id TEXT NOT NULL,
                    collection_name TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    metadata_json TEXT NOT NULL,
                    modality TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, collection_name, item_id),
                    FOREIGN KEY (tenant_id, collection_name)
                        REFERENCES collections (tenant_id, name) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_vectors_scope
                    ON vectors (tenant_id, collection_name);
                CREATE INDEX IF NOT EXISTS idx_vectors_hash
                    ON vectors (tenant_id, collection_name, content_hash);
                """
            )

    def create_collection(self, tenant_id: str, name: str) -> CollectionInfo:
        now = datetime.now(UTC).isoformat()
        with self._lock, self._db:
            self._db.execute(
                """
                INSERT INTO collections (tenant_id, name, dimension, created_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (tenant_id, name) DO NOTHING
                """,
                (tenant_id, name, self.dimension, now),
            )
        return self.get_collection(tenant_id, name)

    def get_collection(self, tenant_id: str, name: str) -> CollectionInfo:
        row = self._db.execute(
            """
            SELECT c.*, COUNT(v.item_id) AS size
            FROM collections c
            LEFT JOIN vectors v
              ON v.tenant_id = c.tenant_id AND v.collection_name = c.name
            WHERE c.tenant_id = ? AND c.name = ?
            GROUP BY c.tenant_id, c.name
            """,
            (tenant_id, name),
        ).fetchone()
        if row is None:
            raise KeyError(name)
        return CollectionInfo(
            name=row["name"],
            tenant_id=row["tenant_id"],
            dimension=row["dimension"],
            size=row["size"],
            created_at=row["created_at"],
        )

    def list_collections(self, tenant_id: str) -> list[CollectionInfo]:
        rows = self._db.execute(
            """
            SELECT c.*, COUNT(v.item_id) AS size
            FROM collections c
            LEFT JOIN vectors v
              ON v.tenant_id = c.tenant_id AND v.collection_name = c.name
            WHERE c.tenant_id = ?
            GROUP BY c.tenant_id, c.name
            ORDER BY c.name
            """,
            (tenant_id,),
        ).fetchall()
        return [
            CollectionInfo(
                name=row["name"],
                tenant_id=row["tenant_id"],
                dimension=row["dimension"],
                size=row["size"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def delete_collection(self, tenant_id: str, name: str) -> bool:
        with self._lock, self._db:
            self._db.execute(
                "DELETE FROM vectors WHERE tenant_id = ? AND collection_name = ?",
                (tenant_id, name),
            )
            cursor = self._db.execute(
                "DELETE FROM collections WHERE tenant_id = ? AND name = ?",
                (tenant_id, name),
            )
        return cursor.rowcount > 0

    @staticmethod
    def _pack(vector: tuple[float, ...]) -> bytes:
        return array("f", vector).tobytes()

    @staticmethod
    def _unpack(payload: bytes) -> tuple[float, ...]:
        values = array("f")
        values.frombytes(payload)
        return tuple(values)

    def upsert(
        self,
        tenant_id: str,
        collection: str,
        records: list[IndexRecord],
        modality: str,
    ) -> int:
        self.create_collection(tenant_id, collection)
        if any(len(record.vector) != self.dimension for record in records):
            raise ValueError(f"All vectors must have dimension {self.dimension}")
        current_size = self.get_collection(tenant_id, collection).size
        existing = {
            row["item_id"]
            for row in self._db.execute(
                """
                SELECT item_id FROM vectors
                WHERE tenant_id = ? AND collection_name = ?
                  AND item_id IN ({})
                """.format(",".join("?" for _ in records)),
                (tenant_id, collection, *(record.item_id for record in records)),
            ).fetchall()
        }
        if current_size + len(records) - len(existing) > self.capacity:
            raise ValueError(f"Collection capacity of {self.capacity} would be exceeded")
        now = datetime.now(UTC).isoformat()
        rows = [
            (
                tenant_id,
                collection,
                record.item_id,
                self._pack(record.vector),
                json.dumps(record.metadata, ensure_ascii=False, separators=(",", ":")),
                modality,
                hashlib.sha256(self._pack(record.vector)).hexdigest(),
                now,
            )
            for record in records
        ]
        with self._lock, self._db:
            self._db.executemany(
                """
                INSERT INTO vectors (
                    tenant_id, collection_name, item_id, vector, metadata_json,
                    modality, content_hash, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (tenant_id, collection_name, item_id) DO UPDATE SET
                    vector = excluded.vector,
                    metadata_json = excluded.metadata_json,
                    modality = excluded.modality,
                    content_hash = excluded.content_hash,
                    updated_at = excluded.updated_at
                """,
                rows,
            )
        return len(records)

    def delete(self, tenant_id: str, collection: str, item_ids: list[str]) -> int:
        placeholders = ",".join("?" for _ in item_ids)
        with self._lock, self._db:
            cursor = self._db.execute(
                f"""
                DELETE FROM vectors
                WHERE tenant_id = ? AND collection_name = ?
                  AND item_id IN ({placeholders})
                """,
                (tenant_id, collection, *item_ids),
            )
        return cursor.rowcount

    def search(
        self,
        tenant_id: str,
        collection: str,
        vector: list[float],
        limit: int,
        metadata_filter: dict[str, Any] | None = None,
        modality: str | None = None,
    ) -> list[tuple[IndexRecord, float, str]]:
        if len(vector) != self.dimension:
            raise ValueError("Query vector has the wrong dimension")
        query = """
            SELECT item_id, vector, metadata_json, modality
            FROM vectors
            WHERE tenant_id = ? AND collection_name = ?
        """
        params: list[Any] = [tenant_id, collection]
        if modality:
            query += " AND modality = ?"
            params.append(modality)
        rows = self._db.execute(query, params).fetchall()
        query_norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        scored: list[tuple[float, str, IndexRecord, str]] = []
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            if metadata_filter and not all(
                metadata.get(key) == value for key, value in metadata_filter.items()
            ):
                continue
            candidate = self._unpack(row["vector"])
            dot = sum(left * right for left, right in zip(vector, candidate, strict=True))
            norm = math.sqrt(sum(value * value for value in candidate)) or 1.0
            score = dot / (query_norm * norm)
            record = IndexRecord(row["item_id"], candidate, metadata)
            scored.append((score, row["item_id"], record, row["modality"]))
        best = heapq.nlargest(limit, scored, key=lambda item: (item[0], item[1]))
        return [(record, score, item_modality) for score, _, record, item_modality in best]

    def count(self, tenant_id: str | None = None) -> int:
        if tenant_id is None:
            row = self._db.execute("SELECT COUNT(*) AS count FROM vectors").fetchone()
        else:
            row = self._db.execute(
                "SELECT COUNT(*) AS count FROM vectors WHERE tenant_id = ?", (tenant_id,)
            ).fetchone()
        return int(row["count"])

    def close(self) -> None:
        with self._lock:
            self._db.close()
