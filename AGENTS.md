# AGENTS.md

## Cursor Cloud specific instructions

This repo is an **AI-powered maths homework worksheet generator**. It has two runnable
services plus one external dependency (Google Gemini):

- **Backend** — FastAPI + Uvicorn (Python). Serves `/api/health` and `/api/worksheets`
  job endpoints and runs the LLM generation via `Usecases/HW_worksheet_generator.py`.
  Job state is in-memory only (no database).
- **Frontend** — React 19 + Vite (TypeScript) SPA that talks to the backend.

Standard install/run/build commands live in `backend/requirements.txt`,
`frontend/package.json` scripts, and `frontend/vite.config.ts`. The dependency-refresh
happens automatically via the startup update script, so you usually only need to start
the two services.

### Running the services

- Python deps are installed into a virtualenv at `/workspace/.venv` (gitignored). Use
  `/workspace/.venv/bin/python` / `/workspace/.venv/bin/uvicorn` (do NOT rely on system
  Python, which is PEP 668 externally-managed).
- **Backend** (port 8000) must be started from the `backend/` directory so the `app.*`
  imports resolve:
  `cd backend && /workspace/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`
- **Frontend** (port 5173): `npm run dev` in `frontend/`. Vite proxies `/api` to
  `http://127.0.0.1:8000`, so start the backend first.

### Non-obvious gotchas

- **Vite dev server binds to IPv6 `localhost` only.** Open/curl `http://localhost:5173`,
  NOT `http://127.0.0.1:5173` (the latter returns a connection error / `000`).
- **`GENAI_API_KEY` is required for real generation.** The backend and generator load it
  from `/workspace/.env` (create `.env` with `GENAI_API_KEY=...`). Without it the app,
  API, form submission, job creation, and background pipeline all still run — but every
  generation job ends in `status: failed` with
  `"No API key was provided..."`. This is expected when the secret is absent, not a
  code/setup bug.
- Generated worksheets are written to `/workspace/AI_output/` (gitignored).

### Lint / test / build

- Backend: no linter or automated test suite is configured. Sanity-check with
  `/workspace/.venv/bin/python -m compileall -q backend/app` and
  `cd backend && /workspace/.venv/bin/python -c "from app.main import app"`.
- Frontend: `npm run build` in `frontend/` runs the `tsc --noEmit` typecheck plus a
  production `vite build` (there is no separate lint or test script).
