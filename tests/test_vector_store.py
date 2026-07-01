from pathlib import Path

from app.services.index import IndexRecord
from app.services.vector_store import SQLiteVectorStore


def test_store_survives_restart_and_isolates_tenants(tmp_path: Path) -> None:
    path = str(tmp_path / "vectors.db")
    first = SQLiteVectorStore(path, dimension=2, capacity=10)
    first.upsert("alpha", "catalog", [IndexRecord("one", (1.0, 0.0), {"v": 1})], "text")
    first.close()

    second = SQLiteVectorStore(path, dimension=2, capacity=10)
    assert second.count("alpha") == 1
    assert second.count("beta") == 0
    result = second.search("alpha", "catalog", [1.0, 0.0], 1)
    assert result[0][0].item_id == "one"
    second.close()
