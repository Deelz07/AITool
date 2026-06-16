import { useCallback, useEffect, useRef, useState } from "react";
import { createWorksheetJob, getWorksheetJob } from "../api/worksheets";
import type { WorksheetFormState, WorksheetJob } from "../types/worksheet";
import { toWorksheetRequest } from "../types/worksheet";

const POLL_INTERVAL_MS = 3000;

interface UseWorksheetGenerationResult {
  job: WorksheetJob | null;
  isSubmitting: boolean;
  error: string | null;
  submit: (form: WorksheetFormState) => Promise<void>;
  reset: () => void;
}

export function useWorksheetGeneration(): UseWorksheetGenerationResult {
  const [job, setJob] = useState<WorksheetJob | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const reset = useCallback(() => {
    stopPolling();
    setJob(null);
    setError(null);
    setIsSubmitting(false);
  }, [stopPolling]);

  const startPolling = useCallback(
    (jobId: string) => {
      stopPolling();
      pollRef.current = window.setInterval(async () => {
        try {
          const updated = await getWorksheetJob(jobId);
          setJob(updated);

          if (updated.status === "completed" || updated.status === "failed") {
            stopPolling();
            setIsSubmitting(false);
            if (updated.status === "failed") {
              setError(updated.error ?? "Generation failed.");
            }
          }
        } catch (pollError) {
          stopPolling();
          setIsSubmitting(false);
          setError(
            pollError instanceof Error
              ? pollError.message
              : "Failed to poll job status.",
          );
        }
      }, POLL_INTERVAL_MS);
    },
    [stopPolling],
  );

  const submit = useCallback(
    async (form: WorksheetFormState) => {
      reset();
      setIsSubmitting(true);

      try {
        const created = await createWorksheetJob(toWorksheetRequest(form));
        setJob(created);
        startPolling(created.id);
      } catch (submitError) {
        setIsSubmitting(false);
        setError(
          submitError instanceof Error
            ? submitError.message
            : "Failed to start generation.",
        );
      }
    },
    [reset, startPolling],
  );

  useEffect(() => stopPolling, [stopPolling]);

  return { job, isSubmitting, error, submit, reset };
}
