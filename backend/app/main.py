from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import PROJECT_ROOT
from app.routes.worksheets import router as worksheets_router

load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(
    title="Worksheet Generator API",
    description="Generate maths homework worksheets from topics and subtopics.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(worksheets_router)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
