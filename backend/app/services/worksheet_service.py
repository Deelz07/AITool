import sys
from pathlib import Path

from app.config import OUTPUT_DIR, PROJECT_ROOT
from app.jobs.store import job_store
from app.schemas.worksheet import WorksheetOutput, WorksheetRequest

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _read_if_exists(path: Path) -> str | None:
    if path.exists():
        return path.read_text(encoding="utf-8")
    return None


def run_generation(job_id: str, request: WorksheetRequest) -> None:
    job_store.update_status(job_id, "running")
    try:
        from Usecases.HW_worksheet_generator import generate_worksheets, topic_paths

        subtopics = [s.strip() for s in request.subtopics if s.strip()]
        if not subtopics:
            raise ValueError("At least one non-empty subtopic is required.")

        results = generate_worksheets(
            topic=request.topic.strip(),
            subtopics=subtopics,
            output_dir=OUTPUT_DIR,
            core_per_subtopic=request.core_questions_per_subtopic,
            challenge_per_subtopic=request.challenge_questions_per_subtopic,
            tech_active_per_subtopic=request.tech_active_questions_per_subtopic,
            contest_level=request.contest_level,
            diagram_mode=request.diagram_mode,
            diagrams_per_subtopic=request.diagrams_per_subtopic,
            step=request.step,
        )

        paths = topic_paths(request.topic.strip(), OUTPUT_DIR)
        outputs = WorksheetOutput(
            research=_read_if_exists(results.get("research", paths["research"])),
            worksheet=_read_if_exists(results.get("worksheet", paths["worksheet"])),
            solutions=_read_if_exists(results.get("solutions", paths["solutions"])),
        )
        job_store.set_outputs(job_id, outputs)
    except Exception as exc:
        job_store.update_status(job_id, "failed", str(exc))
