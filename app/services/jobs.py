import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

JobState = Literal["queued", "running", "succeeded", "failed"]


@dataclass(slots=True)
class JobRecord:
    id: str
    kind: str
    tenant_id: str
    state: JobState = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class JobManager:
    """Bounded in-process job executor.

    This provider gives local deployments a complete asynchronous workflow.
    Production clusters can swap it for Celery/Arq/Kafka behind the same API.
    """

    def __init__(self, workers: int, max_history: int) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=workers,
            thread_name_prefix="clipforge-job",
        )
        self._jobs: dict[str, JobRecord] = {}
        self._lock = threading.RLock()
        self._max_history = max_history

    def submit(
        self,
        kind: str,
        tenant_id: str,
        operation: Callable[[], dict[str, Any]],
    ) -> JobRecord:
        job = JobRecord(id=str(uuid.uuid4()), kind=kind, tenant_id=tenant_id)
        with self._lock:
            if len(self._jobs) >= self._max_history:
                completed = [
                    item for item in self._jobs.values() if item.state in {"succeeded", "failed"}
                ]
                if completed:
                    oldest = min(completed, key=lambda item: item.created_at)
                    self._jobs.pop(oldest.id, None)
            self._jobs[job.id] = job
        self._executor.submit(self._run, job.id, operation)
        return job

    def _run(self, job_id: str, operation: Callable[[], dict[str, Any]]) -> None:
        with self._lock:
            job = self._jobs[job_id]
            job.state = "running"
            job.started_at = datetime.now(UTC).isoformat()
        try:
            result = operation()
        except Exception as exc:
            with self._lock:
                job.state = "failed"
                job.error = f"{type(exc).__name__}: {exc}"
                job.finished_at = datetime.now(UTC).isoformat()
        else:
            with self._lock:
                job.state = "succeeded"
                job.result = result
                job.finished_at = datetime.now(UTC).isoformat()

    def get(self, job_id: str, tenant_id: str) -> JobRecord:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.tenant_id != tenant_id:
                raise KeyError(job_id)
            return job

    def list(self, tenant_id: str, limit: int = 50) -> list[JobRecord]:
        with self._lock:
            jobs = [job for job in self._jobs.values() if job.tenant_id == tenant_id]
        return sorted(jobs, key=lambda item: item.created_at, reverse=True)[:limit]

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)
