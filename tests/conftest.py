import os

os.environ["CLIPFORGE_MODEL_BACKEND"] = "mock"
os.environ["CLIPFORGE_EMBEDDING_DIMENSION"] = "32"
os.environ["CLIPFORGE_API_KEYS"] = '["test-key"]'
os.environ["CLIPFORGE_VECTOR_STORE_PATH"] = ":memory:"

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth() -> dict[str, str]:
    return {"X-API-Key": "test-key"}
