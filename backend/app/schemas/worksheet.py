from typing import Literal

from pydantic import BaseModel, Field


ContestLevel = Literal["AMC 10", "AMC 12", "AIME", "UKMT Senior", "IMO"]
GenerationStep = Literal["research", "questions", "solutions", "all"]
JobStatus = Literal["queued", "running", "completed", "failed"]


class WorksheetRequest(BaseModel):
    topic: str = Field(..., min_length=1, max_length=200)
    subtopics: list[str] = Field(..., min_length=1)
    step: GenerationStep = "all"
    core_questions_per_subtopic: int = Field(default=5, ge=1, le=20)
    challenge_questions_per_subtopic: int = Field(default=3, ge=0, le=20)
    tech_active_questions_per_subtopic: int = Field(default=1, ge=0, le=10)
    contest_level: ContestLevel = "IMO"
    diagram_mode: bool = False
    diagrams_per_subtopic: int = Field(default=1, ge=0, le=5)


class WorksheetOutput(BaseModel):
    research: str | None = None
    worksheet: str | None = None
    solutions: str | None = None


class WorksheetJob(BaseModel):
    id: str
    status: JobStatus
    topic: str
    step: GenerationStep
    error: str | None = None
    outputs: WorksheetOutput | None = None
    created_at: str
    updated_at: str
