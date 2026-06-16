from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.jobs.store import job_store
from app.schemas.worksheet import WorksheetJob, WorksheetRequest
from app.services.worksheet_service import run_generation

router = APIRouter(prefix="/api/worksheets", tags=["worksheets"])


@router.post("", response_model=WorksheetJob, status_code=202)
def create_worksheet_job(
    request: WorksheetRequest,
    background_tasks: BackgroundTasks,
) -> WorksheetJob:
    job = job_store.create(request)
    background_tasks.add_task(run_generation, job.id, request)
    return job


@router.get("/{job_id}", response_model=WorksheetJob)
def get_worksheet_job(job_id: str) -> WorksheetJob:
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job
