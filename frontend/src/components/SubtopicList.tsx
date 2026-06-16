interface SubtopicListProps {
  subtopics: string[];
  disabled?: boolean;
  onChange: (subtopics: string[]) => void;
}

export function SubtopicList({ subtopics, disabled, onChange }: SubtopicListProps) {
  const updateSubtopic = (index: number, value: string) => {
    const next = [...subtopics];
    next[index] = value;
    onChange(next);
  };

  const addSubtopic = () => {
    onChange([...subtopics, ""]);
  };

  const removeSubtopic = (index: number) => {
    if (subtopics.length === 1) {
      onChange([""]);
      return;
    }
    onChange(subtopics.filter((_, i) => i !== index));
  };

  return (
    <div className="subtopic-list">
      <div className="field-header">
        <label>Subtopics</label>
        <span className="field-hint">One skill or concept per line</span>
      </div>

      <div className="subtopic-rows">
        {subtopics.map((subtopic, index) => (
          <div className="subtopic-row" key={index}>
            <span className="subtopic-index">{index + 1}</span>
            <input
              type="text"
              value={subtopic}
              disabled={disabled}
              placeholder="e.g. Prime factorisation"
              onChange={(event) => updateSubtopic(index, event.target.value)}
            />
            <button
              type="button"
              className="icon-button"
              disabled={disabled}
              aria-label={`Remove subtopic ${index + 1}`}
              onClick={() => removeSubtopic(index)}
            >
              ×
            </button>
          </div>
        ))}
      </div>

      <button
        type="button"
        className="text-button"
        disabled={disabled}
        onClick={addSubtopic}
      >
        + Add subtopic
      </button>
    </div>
  );
}
