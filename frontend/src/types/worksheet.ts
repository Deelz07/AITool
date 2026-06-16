export type ContestLevel =
  | "AMC 10"
  | "AMC 12"
  | "AIME"
  | "UKMT Senior"
  | "IMO";

export type GenerationStep = "research" | "questions" | "solutions" | "all";

export type JobStatus = "queued" | "running" | "completed" | "failed";

export interface WorksheetRequest {
  topic: string;
  subtopics: string[];
  step: GenerationStep;
  core_questions_per_subtopic: number;
  challenge_questions_per_subtopic: number;
  tech_active_questions_per_subtopic: number;
  contest_level: ContestLevel;
  diagram_mode: boolean;
  diagrams_per_subtopic: number;
}

export interface WorksheetOutput {
  research: string | null;
  worksheet: string | null;
  solutions: string | null;
}

export interface WorksheetJob {
  id: string;
  status: JobStatus;
  topic: string;
  step: GenerationStep;
  error: string | null;
  outputs: WorksheetOutput | null;
  created_at: string;
  updated_at: string;
}

export interface WorksheetFormState {
  topic: string;
  subtopics: string[];
  step: GenerationStep;
  coreQuestionsPerSubtopic: number;
  challengeQuestionsPerSubtopic: number;
  techActiveQuestionsPerSubtopic: number;
  contestLevel: ContestLevel;
  diagramMode: boolean;
  diagramsPerSubtopic: number;
}

export const defaultFormState: WorksheetFormState = {
  topic: "",
  subtopics: [""],
  step: "all",
  coreQuestionsPerSubtopic: 5,
  challengeQuestionsPerSubtopic: 3,
  techActiveQuestionsPerSubtopic: 1,
  contestLevel: "IMO",
  diagramMode: false,
  diagramsPerSubtopic: 1,
};

export function toWorksheetRequest(form: WorksheetFormState): WorksheetRequest {
  return {
    topic: form.topic.trim(),
    subtopics: form.subtopics.map((s) => s.trim()).filter(Boolean),
    step: form.step,
    core_questions_per_subtopic: form.coreQuestionsPerSubtopic,
    challenge_questions_per_subtopic: form.challengeQuestionsPerSubtopic,
    tech_active_questions_per_subtopic: form.techActiveQuestionsPerSubtopic,
    contest_level: form.contestLevel,
    diagram_mode: form.diagramMode,
    diagrams_per_subtopic: form.diagramsPerSubtopic,
  };
}
