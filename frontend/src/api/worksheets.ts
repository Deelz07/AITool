import type { WorksheetJob, WorksheetRequest } from "../types/worksheet";

const API_BASE = "/api";

async function parseError(response: Response): Promise<string> {
  try {
    const data = (await response.json()) as { detail?: string | { msg: string }[] };
    if (typeof data.detail === "string") return data.detail;
    if (Array.isArray(data.detail)) {
      return data.detail.map((item) => item.msg).join(", ");
    }
  } catch {
    // ignore JSON parse errors
  }
  return `Request failed (${response.status})`;
}

export async function createWorksheetJob(
  request: WorksheetRequest,
): Promise<WorksheetJob> {
  const response = await fetch(`${API_BASE}/worksheets`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return response.json() as Promise<WorksheetJob>;
}

export async function getWorksheetJob(jobId: string): Promise<WorksheetJob> {
  const response = await fetch(`${API_BASE}/worksheets/${jobId}`);

  if (!response.ok) {
    throw new Error(await parseError(response));
  }

  return response.json() as Promise<WorksheetJob>;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const response = await fetch(`${API_BASE}/health`);
    return response.ok;
  } catch {
    return false;
  }
}
