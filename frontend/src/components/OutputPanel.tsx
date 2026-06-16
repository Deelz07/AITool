import { useMemo, useState } from "react";
import type { WorksheetOutput } from "../types/worksheet";

interface OutputPanelProps {
  topic: string;
  outputs: WorksheetOutput;
  onReset: () => void;
}

type OutputTab = "worksheet" | "solutions" | "research";

function downloadText(filename: string, content: string) {
  const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  URL.revokeObjectURL(url);
}

export function OutputPanel({ topic, outputs, onReset }: OutputPanelProps) {
  const tabs = useMemo(() => {
    const available: { id: OutputTab; label: string; content: string }[] = [];
    if (outputs.worksheet) {
      available.push({ id: "worksheet", label: "Worksheet (.tex)", content: outputs.worksheet });
    }
    if (outputs.solutions) {
      available.push({ id: "solutions", label: "Solutions (.tex)", content: outputs.solutions });
    }
    if (outputs.research) {
      available.push({ id: "research", label: "Research notes", content: outputs.research });
    }
    return available;
  }, [outputs]);

  const [activeTab, setActiveTab] = useState<OutputTab>(
    tabs[0]?.id ?? "worksheet",
  );

  const activeContent = tabs.find((tab) => tab.id === activeTab)?.content ?? "";

  const slug = topic
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_|_$/g, "");

  return (
    <section className="output-panel">
      <div className="output-header">
        <div>
          <h2>Your worksheet is ready</h2>
          <p>Download the LaTeX files or preview the generated content below.</p>
        </div>
        <button type="button" className="secondary-button" onClick={onReset}>
          Create another
        </button>
      </div>

      <div className="output-actions">
        {outputs.worksheet && (
          <button
            type="button"
            className="text-button"
            onClick={() =>
              downloadText(`${slug}_worksheet.tex`, outputs.worksheet!)
            }
          >
            Download worksheet
          </button>
        )}
        {outputs.solutions && (
          <button
            type="button"
            className="text-button"
            onClick={() =>
              downloadText(`${slug}_solutions.tex`, outputs.solutions!)
            }
          >
            Download solutions
          </button>
        )}
        {outputs.research && (
          <button
            type="button"
            className="text-button"
            onClick={() =>
              downloadText(`${slug}_research.txt`, outputs.research!)
            }
          >
            Download research
          </button>
        )}
      </div>

      {tabs.length > 0 && (
        <>
          <div className="output-tabs" role="tablist">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                type="button"
                role="tab"
                aria-selected={activeTab === tab.id}
                className={activeTab === tab.id ? "active" : ""}
                onClick={() => setActiveTab(tab.id)}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <pre className="output-preview">{activeContent}</pre>
        </>
      )}
    </section>
  );
}
