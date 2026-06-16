from datetime import datetime, timezone
from threading import Lock
from uuid import uuid4

from app.schemas.worksheet import JobStatus, WorksheetJob, WorksheetOutput, WorksheetRequest


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, WorksheetJob] = {}
        self._lock = Lock()

    def create(self, request: WorksheetRequest) -> WorksheetJob:
        job = WorksheetJob(
            id=str(uuid4()),
            status="queued",
            topic=request.topic.strip(),
            step=request.step,
            created_at=_utc_now(),
            updated_at=_utc_now(),
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> WorksheetJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update_status(self, job_id: str, status: JobStatus, error: str | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = status
            job.error = error
            job.updated_at = _utc_now()

    def set_outputs(self, job_id: str, outputs: WorksheetOutput) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.outputs = outputs
            job.status = "completed"
            job.updated_at = _utc_now()


job_store = JobStore()
