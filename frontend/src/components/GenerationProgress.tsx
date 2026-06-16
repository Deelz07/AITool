import type { JobStatus } from "../types/worksheet";

interface GenerationProgressProps {
  status: JobStatus | null;
  topic: string;
}

const STATUS_LABELS: Record<JobStatus, string> = {
  queued: "Queued",
  running: "Generating",
  completed: "Complete",
  failed: "Failed",
};

const STATUS_STEPS: JobStatus[] = ["queued", "running", "completed"];

export function GenerationProgress({ status, topic }: GenerationProgressProps) {
  if (!status) return null;

  const activeIndex = STATUS_STEPS.indexOf(
    status === "failed" ? "running" : status,
  );

  return (
    <section className="progress-panel">
      <div className="progress-header">
        <h2>Generation in progress</h2>
        <span className={`status-badge status-${status}`}>
          {STATUS_LABELS[status]}
        </span>
      </div>

      <p className="progress-topic">
        Topic: <strong>{topic}</strong>
      </p>

      <div className="progress-steps">
        {STATUS_STEPS.map((step, index) => (
          <div
            key={step}
            className={`progress-step ${
              index <= activeIndex ? "active" : ""
            } ${status === "failed" && step === "running" ? "failed" : ""}`}
          >
            <span className="step-dot" />
            <span>{STATUS_LABELS[step]}</span>
          </div>
        ))}
      </div>

      {(status === "queued" || status === "running") && (
        <p className="progress-note">
          This can take several minutes. Research, question writing, and solutions
          each call the AI model separately.
        </p>
      )}
    </section>
  );
}
