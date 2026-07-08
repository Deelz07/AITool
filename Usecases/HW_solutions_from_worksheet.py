"""
Generate worked solutions from an existing worksheet .tex file.

This skips research and question generation — it only reads the worksheet you
configure below and writes a matching solutions file.

Edit WORKSHEET_PATH (and optional settings) then run:
  python3 -m Usecases.HW_solutions_from_worksheet
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import generate_content as AI

# --- Edit these before running ---
WORKSHEET_PATH = PROJECT_ROOT / "AI_output/discrete_time_stochastic_processes_worksheet.tex"
OUTPUT_PATH = None  # default: alongside worksheet as *_solutions.tex
TOPIC = None  # inferred from worksheet title if None
SUBTOPICS = None  # inferred from Section A subsections if None
CORE_QUESTIONS_PER_SUBTOPIC = 4
CHALLENGE_QUESTIONS_PER_SUBTOPIC = 3
TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC = 1
CONTEST_LEVEL = "USAJMO"
DIAGRAM_MODE = False
DIAGRAMS_PER_SUBTOPIC = 1
USE_OG = True  # True: HW_worksheet_generator_og.py | False: split-section engine


def resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    if not resolved.exists():
        raise FileNotFoundError(f"Worksheet not found: {resolved}")
    return resolved


def infer_topic_from_worksheet(worksheet: str, worksheet_path: Path) -> str:
    title_match = re.search(r"\\title\{([^}]+)\}", worksheet)
    if title_match:
        title = title_match.group(1).strip()
        title = re.sub(r"^Worksheet:\s*", "", title, flags=re.IGNORECASE).strip()
        if title:
            return title

    stem = worksheet_path.stem
    if stem.endswith("_worksheet"):
        return stem[: -len("_worksheet")].replace("_", " ").title()
    return stem.replace("_", " ").title()


def extract_section_body(worksheet: str, section_letter: str) -> str:
    pattern = rf"\\section\*?\{{Section {section_letter}[^}}]*\}}(.*?)(\\section\*?\{{|\\newpage|\\end{{document}})"
    match = re.search(pattern, worksheet, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1) if match else ""


def infer_subtopics_from_worksheet(worksheet: str) -> list[str]:
    section_a = extract_section_body(worksheet, "A")
    if not section_a:
        return []

    subtopics = re.findall(r"\\subsection\*?\{([^}]+)\}", section_a)
    return [name.strip() for name in subtopics if name.strip()]


def default_solutions_path(worksheet_path: Path) -> Path:
    name = worksheet_path.name
    if name.endswith("_worksheet.tex"):
        return worksheet_path.with_name(name.replace("_worksheet.tex", "_solutions.tex"))
    return worksheet_path.with_name(f"{worksheet_path.stem}_solutions.tex")


def run_solutions_from_worksheet(
    worksheet_path: Path,
    *,
    output_path: Path | None = None,
    topic: str | None = None,
    subtopics: list[str] | None = None,
    core_per_subtopic: int = CORE_QUESTIONS_PER_SUBTOPIC,
    challenge_per_subtopic: int = CHALLENGE_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level: str = CONTEST_LEVEL,
    diagram_mode: bool = DIAGRAM_MODE,
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
    use_og: bool = USE_OG,
) -> Path:
    if use_og:
        import Usecases.HW_worksheet_generator_og as generator
    else:
        import Usecases.HW_worksheet_generator as generator

    generator.setup_logging()

    worksheet_tex = worksheet_path.read_text(encoding="utf-8")
    resolved_topic = topic or infer_topic_from_worksheet(worksheet_tex, worksheet_path)
    resolved_subtopics = subtopics or infer_subtopics_from_worksheet(worksheet_tex)
    if not resolved_subtopics:
        raise ValueError(
            "Could not infer subtopics from the worksheet. "
            "Set SUBTOPICS in this script or ensure Section A has \\subsection titles."
        )

    solutions_path = output_path or default_solutions_path(worksheet_path)

    generator.logger.info("Worksheet: %s (%d chars)", worksheet_path, len(worksheet_tex))
    generator.logger.info("Topic: %s", resolved_topic)
    generator.logger.info("Subtopics (%d): %s", len(resolved_subtopics), resolved_subtopics)
    generator.logger.info("Solutions output: %s", solutions_path)
    generator.logger.info("Engine: %s", "og" if use_og else "split-section")

    with generator.log_step("solutions"):
        client = AI.setup_client()
        generator.logger.info("Gemini client initialised")
        solutions_tex = generator.generate_solutions_worksheet(
            client,
            resolved_topic,
            resolved_subtopics,
            worksheet_tex,
            core_per_subtopic,
            challenge_per_subtopic,
            tech_active_per_subtopic,
            contest_level,
            diagram_mode,
            diagrams_per_subtopic,
        )
        generator.write_output(solutions_path, solutions_tex)

    return solutions_path


def main() -> None:
    worksheet_path = resolve_path(WORKSHEET_PATH)
    output_path = None
    if OUTPUT_PATH:
        output_path = Path(OUTPUT_PATH).expanduser()
        if not output_path.is_absolute():
            output_path = PROJECT_ROOT / output_path

    solutions_path = run_solutions_from_worksheet(
        worksheet_path,
        output_path=output_path,
        topic=TOPIC,
        subtopics=SUBTOPICS,
        core_per_subtopic=CORE_QUESTIONS_PER_SUBTOPIC,
        challenge_per_subtopic=CHALLENGE_QUESTIONS_PER_SUBTOPIC,
        tech_active_per_subtopic=TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
        contest_level=CONTEST_LEVEL,
        diagram_mode=DIAGRAM_MODE,
        diagrams_per_subtopic=DIAGRAMS_PER_SUBTOPIC,
        use_og=USE_OG,
    )
    print(f"Solutions written to: {solutions_path}")


if __name__ == "__main__":
    main()
