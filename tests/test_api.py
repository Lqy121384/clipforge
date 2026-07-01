import base64
import time

import pytest
from fastapi.testclient import TestClient


def test_health_is_public(client: TestClient) -> None:
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.headers["X-Request-ID"]


def test_protected_endpoint_rejects_missing_key(client: TestClient) -> None:
    response = client.get("/api/v1/model")
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "invalid_api_key"


def test_text_embeddings_are_normalized_and_deterministic(
    client: TestClient, auth: dict[str, str]
) -> None:
    payload = {"texts": ["red running shoes", "red running shoes"]}
    response = client.post("/api/v1/embeddings/text", json=payload, headers=auth)
    assert response.status_code == 200
    body = response.json()
    assert body["dimension"] == 32
    assert body["embeddings"][0] == body["embeddings"][1]
    norm = sum(value**2 for value in body["embeddings"][0]) ** 0.5
    assert norm == pytest.approx(1.0)


def test_similarity_contract(client: TestClient, auth: dict[str, str]) -> None:
    image = base64.b64encode(b"fake-image-contract-payload").decode()
    response = client.post(
        "/api/v1/similarity",
        json={"texts": ["cat", "dog"], "images": [image]},
        headers=auth,
    )
    assert response.status_code == 200
    assert len(response.json()["scores"]) == 2
    assert len(response.json()["scores"][0]) == 1


def test_prompt_ensemble_classification(client: TestClient, auth: dict[str, str]) -> None:
    image = base64.b64encode(b"classification-image").decode()
    response = client.post(
        "/api/v1/classifications/zero-shot",
        json={
            "images": [image],
            "labels": ["product photo", "landscape"],
            "templates": ["a photo of {}.", "an image showing {}."],
            "top_k": 2,
        },
        headers=auth,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["prompt_count"] == 4
    assert len(body["results"][0]["predictions"]) == 2
    probability = sum(item["probability"] for item in body["results"][0]["predictions"])
    assert probability == pytest.approx(1.0)


def test_index_upsert_search_and_delete(client: TestClient, auth: dict[str, str]) -> None:
    items = [
        {"id": "a", "text": "waterproof mountain jacket", "metadata": {"team": "outdoor"}},
        {"id": "b", "text": "formal leather office shoe", "metadata": {"team": "fashion"}},
    ]
    upsert = client.post("/api/v1/index/text", json={"items": items}, headers=auth)
    assert upsert.status_code == 200
    assert upsert.json()["index_size"] == 2

    search = client.post(
        "/api/v1/search/text",
        json={"query": "mountain jacket", "limit": 2, "metadata_filter": {"team": "outdoor"}},
        headers=auth,
    )
    assert search.status_code == 200
    assert [hit["id"] for hit in search.json()["hits"]] == ["a"]

    deleted = client.post("/api/v1/index/delete", json={"ids": ["a"]}, headers=auth)
    assert deleted.status_code == 200
    assert deleted.json()["affected"] == 1


def test_batch_limit(client: TestClient, auth: dict[str, str]) -> None:
    response = client.post(
        "/api/v1/embeddings/text",
        json={"texts": [f"item {index}" for index in range(65)]},
        headers=auth,
    )
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "batch_too_large"


def test_console_is_served(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "CLIPForge" in response.text


def test_collections_are_tenant_isolated(client: TestClient, auth: dict[str, str]) -> None:
    tenant_a = {**auth, "X-Tenant-ID": "acme"}
    tenant_b = {**auth, "X-Tenant-ID": "globex"}
    created = client.post(
        "/api/v1/collections",
        json={"name": "catalog"},
        headers=tenant_a,
    )
    assert created.status_code == 201

    client.post(
        "/api/v1/collections/catalog/items/text",
        json={"items": [{"id": "sku-1", "text": "red running shoe"}]},
        headers=tenant_a,
    )
    acme = client.get("/api/v1/collections", headers=tenant_a).json()
    globex = client.get("/api/v1/collections", headers=tenant_b).json()
    assert acme["collections"][0]["size"] == 1
    assert globex["total"] == 0


def test_cross_modal_collection_search(client: TestClient, auth: dict[str, str]) -> None:
    image = base64.b64encode(b"enterprise-image-payload").decode()
    indexed = client.post(
        "/api/v1/collections/media/items/image",
        json={"items": [{"id": "asset-1", "image": image, "metadata": {"kind": "hero"}}]},
        headers=auth,
    )
    assert indexed.status_code == 200
    response = client.post(
        "/api/v1/collections/media/search",
        json={
            "query_type": "text",
            "query": "website hero image",
            "target_modality": "image",
        },
        headers=auth,
    )
    assert response.status_code == 200
    assert response.json()["hits"][0]["id"] == "asset-1"
    assert response.json()["hits"][0]["modality"] == "image"


def test_async_index_job(client: TestClient, auth: dict[str, str]) -> None:
    submitted = client.post(
        "/api/v1/jobs/index/text",
        json={
            "collection": "bulk",
            "items": [{"id": f"doc-{index}", "text": f"document {index}"} for index in range(5)],
        },
        headers=auth,
    )
    assert submitted.status_code == 202
    job_id = submitted.json()["id"]
    body = submitted.json()
    for _ in range(100):
        body = client.get(f"/api/v1/jobs/{job_id}", headers=auth).json()
        if body["state"] in {"succeeded", "failed"}:
            break
        time.sleep(0.01)
    assert body["state"] == "succeeded"
    assert body["result"]["affected"] == 5


def test_interactive_search_clarifies_and_applies_feedback(
    client: TestClient, auth: dict[str, str]
) -> None:
    items = [
        {"id": "formal", "text": "formal leather office shoe"},
        {"id": "casual", "text": "comfortable casual sneaker"},
        {"id": "outdoor", "text": "waterproof mountain boot"},
    ]
    indexed = client.post(
        "/api/v1/collections/shoes/items/text",
        json={"items": items},
        headers=auth,
    )
    assert indexed.status_code == 200
    initial = client.post(
        "/api/v1/collections/shoes/search/interactive",
        json={
            "query_type": "text",
            "query": "a shoe for an event",
            "limit": 3,
            "margin_threshold": 0.99,
            "entropy_threshold": 0.01,
        },
        headers=auth,
    )
    assert initial.status_code == 200
    initial_body = initial.json()
    assert initial_body["uncertainty"]["needs_clarification"] is True
    assert len(initial_body["clarification"]["options"]) == 2

    positive = initial_body["clarification"]["options"][0]["id"]
    negative = initial_body["clarification"]["options"][1]["id"]
    refined = client.post(
        "/api/v1/collections/shoes/search/interactive",
        json={
            "query_type": "text",
            "query": "a shoe for an event",
            "limit": 3,
            "feedback": {
                "positive_ids": [positive],
                "negative_ids": [negative],
            },
        },
        headers=auth,
    )
    assert refined.status_code == 200
    refined_body = refined.json()
    assert refined_body["feedback_applied"] is True
    assert refined_body["query_drift"] > 0
    assert refined_body["hits"][0]["id"] == positive
