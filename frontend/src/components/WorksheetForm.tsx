import type { WorksheetFormState } from "../types/worksheet";
import { SubtopicList } from "./SubtopicList";

interface WorksheetFormProps {
  form: WorksheetFormState;
  disabled?: boolean;
  onChange: (form: WorksheetFormState) => void;
  onSubmit: () => void;
}

export function WorksheetForm({
  form,
  disabled,
  onChange,
  onSubmit,
}: WorksheetFormProps) {
  const update = <K extends keyof WorksheetFormState>(
    key: K,
    value: WorksheetFormState[K],
  ) => {
    onChange({ ...form, [key]: value });
  };

  const handleSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    onSubmit();
  };

  const hasTopic = form.topic.trim().length > 0;
  const hasSubtopics = form.subtopics.some((s) => s.trim().length > 0);

  return (
    <form className="worksheet-form" onSubmit={handleSubmit}>
      <section className="form-section">
        <h2>Topic</h2>
        <div className="field">
          <label htmlFor="topic">Main topic</label>
          <input
            id="topic"
            type="text"
            value={form.topic}
            disabled={disabled}
            placeholder="e.g. Number theory, ETC3430 Week 1"
            onChange={(event) => update("topic", event.target.value)}
          />
        </div>

        <SubtopicList
          subtopics={form.subtopics}
          disabled={disabled}
          onChange={(subtopics) => update("subtopics", subtopics)}
        />
      </section>

      <details className="advanced-options">
        <summary>Advanced options</summary>

        <div className="options-grid">
          <div className="field">
            <label htmlFor="step">Generation step</label>
            <select
              id="step"
              value={form.step}
              disabled={disabled}
              onChange={(event) =>
                update("step", event.target.value as WorksheetFormState["step"])
              }
            >
              <option value="all">All (research → questions → solutions)</option>
              <option value="research">Research only</option>
              <option value="questions">Questions only</option>
              <option value="solutions">Solutions only</option>
            </select>
          </div>

          <div className="field">
            <label htmlFor="contestLevel">Challenge difficulty</label>
            <select
              id="contestLevel"
              value={form.contestLevel}
              disabled={disabled}
              onChange={(event) =>
                update(
                  "contestLevel",
                  event.target.value as WorksheetFormState["contestLevel"],
                )
              }
            >
              <option value="AMC 10">AMC 10</option>
              <option value="AMC 12">AMC 12</option>
              <option value="AIME">AIME</option>
              <option value="UKMT Senior">UKMT Senior</option>
              <option value="IMO">IMO</option>
            </select>
          </div>

          <div className="field">
            <label htmlFor="coreQuestions">Core questions per subtopic</label>
            <input
              id="coreQuestions"
              type="number"
              min={1}
              max={20}
              value={form.coreQuestionsPerSubtopic}
              disabled={disabled}
              onChange={(event) =>
                update("coreQuestionsPerSubtopic", Number(event.target.value))
              }
            />
          </div>

          <div className="field">
            <label htmlFor="challengeQuestions">Challenge questions per subtopic</label>
            <input
              id="challengeQuestions"
              type="number"
              min={0}
              max={20}
              value={form.challengeQuestionsPerSubtopic}
              disabled={disabled}
              onChange={(event) =>
                update("challengeQuestionsPerSubtopic", Number(event.target.value))
              }
            />
          </div>

          <div className="field">
            <label htmlFor="techActiveQuestions">Tech-active questions per subtopic</label>
            <input
              id="techActiveQuestions"
              type="number"
              min={0}
              max={10}
              value={form.techActiveQuestionsPerSubtopic}
              disabled={disabled}
              onChange={(event) =>
                update("techActiveQuestionsPerSubtopic", Number(event.target.value))
              }
            />
          </div>

          <div className="field checkbox-field">
            <label htmlFor="diagramMode">
              <input
                id="diagramMode"
                type="checkbox"
                checked={form.diagramMode}
                disabled={disabled}
                onChange={(event) => update("diagramMode", event.target.checked)}
              />
              Enable diagram mode (TikZ figures)
            </label>
          </div>

          {form.diagramMode && (
            <div className="field">
              <label htmlFor="diagramsPerSubtopic">Diagrams per subtopic</label>
              <input
                id="diagramsPerSubtopic"
                type="number"
                min={0}
                max={5}
                value={form.diagramsPerSubtopic}
                disabled={disabled}
                onChange={(event) =>
                  update("diagramsPerSubtopic", Number(event.target.value))
                }
              />
            </div>
          )}
        </div>
      </details>

      <button
        type="submit"
        className="primary-button"
        disabled={disabled || !hasTopic || !hasSubtopics}
      >
        Generate worksheet
      </button>
    </form>
  );
}
