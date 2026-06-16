import { useEffect, useState } from "react";
import { checkHealth } from "./api/worksheets";
import { GenerationProgress } from "./components/GenerationProgress";
import { OutputPanel } from "./components/OutputPanel";
import { WorksheetForm } from "./components/WorksheetForm";
import { useWorksheetGeneration } from "./hooks/useWorksheetGeneration";
import { defaultFormState } from "./types/worksheet";
import type { WorksheetFormState } from "./types/worksheet";
import "./styles/index.css";

export default function App() {
  const [form, setForm] = useState<WorksheetFormState>(defaultFormState);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const { job, isSubmitting, error, submit, reset } = useWorksheetGeneration();

  useEffect(() => {
    checkHealth().then(setBackendOnline);
  }, []);

  const handleSubmit = () => {
    void submit(form);
  };

  const showForm = !job || job.status === "failed";
  const showProgress =
    job && (job.status === "queued" || job.status === "running");

  return (
    <div className="app">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark">Σ</span>
          <div>
            <h1>Worksheet Generator</h1>
            <p>Research-backed maths homework from your topics</p>
          </div>
        </div>

        <div className="header-status">
          {backendOnline === false && (
            <span className="offline-badge">Backend offline</span>
          )}
          {backendOnline === true && (
            <span className="online-badge">API connected</span>
          )}
        </div>
      </header>

      <main className="app-main">
        {backendOnline === false && (
          <div className="alert alert-warning">
            Start the backend with{" "}
            <code>
              uvicorn app.main:app --reload --app-dir backend
            </code>{" "}
            before generating worksheets.
          </div>
        )}

        {error && (
          <div className="alert alert-error" role="alert">
            {error}
          </div>
        )}

        {showForm && (
          <WorksheetForm
            form={form}
            disabled={isSubmitting}
            onChange={setForm}
            onSubmit={handleSubmit}
          />
        )}

        {showProgress && job && (
          <GenerationProgress status={job.status} topic={job.topic} />
        )}

        {job?.status === "completed" && job.outputs && (
          <OutputPanel
            topic={job.topic}
            outputs={job.outputs}
            onReset={reset}
          />
        )}
      </main>
    </div>
  );
}
