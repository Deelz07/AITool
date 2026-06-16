"""
Generate maths homework worksheets from a topic and subtopics by researching
reputable online resources, then producing a LaTeX question sheet and a matching
solutions sheet derived from that worksheet.

LEGACY VERSION: single API call for solutions (before split-section architecture).
Use HW_worksheet_generator.py for the current multi-call implementation.

Each LLM call can be run separately via the STEP setting. Intermediate files
are saved under AI_output/ so a failed step can be retried without redoing earlier work.
"""

import logging
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import generate_content as AI
from google.genai import types

# Flash-Lite: lower demand / higher availability than gemini-2.5-flash
MODEL = "gemini-2.5-flash"
OUTPUT_DIR = PROJECT_ROOT / "AI_output"
PAST_QUERIES_FILE = PROJECT_ROOT / "past queries.txt"
LOG_FILE = OUTPUT_DIR / "hw_generator_og.log"

logger = logging.getLogger("hw_worksheet")

# --- Edit these before running ---
TOPIC = "Markov Jump processes"

SUBTOPICS= [ "Assumptions behind Markov chains and Markov jump processes",
    "Generator matrices for Markov jump processes",
    "Solving Kolmogorov differential equations to find probabilities",
    "Limiting distribution for Markov jump processes",
    "Stationary distribution for Markov jump processes",
    "Poisson process"
]

CORE_QUESTIONS_PER_SUBTOPIC = 4
CHALLENGE_QUESTIONS_PER_SUBTOPIC = 3
TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC = 1
# Target difficulty for challenge section: AMC 10, AMC 12, AIME, or UKMT Senior
CONTEST_LEVEL = "USAJMO"
# Run one step at a time: "research", "questions", "solutions", or "all"
STEP = "all"
# Set True for geometry/visual topics (uses TikZ diagrams in LaTeX output)
DIAGRAM_MODE = False
DIAGRAMS_PER_SUBTOPIC = 1  # minimum diagram-backed questions per subtopic when DIAGRAM_MODE is True
# Use Gemini Batch API for 50% token discount (async — may take minutes per call)
USE_BATCH_API = False
BATCH_POLL_INTERVAL_SEC = 30

BATCH_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

REPUTABLE_SOURCES = """
Prioritise content from reputable educational sources such as:
- Art of Problem Solving (AOPS), Khan Academy, OpenStax, BBC Bitesize, Paul's Online Math Notes
- MIT OpenCourseWare, university lecture notes, and official exam-board specs (AQA, Edexcel, OCR)
- Wolfram MathWorld for definitions and standard techniques
For challenge problems, also study the style of AMC 10/12, AIME, UKMT, USAJMO,IMO and HMMT —
but write original contest-style problems; never copy existing contest questions.
For tech-active problems, draw on extended-response CAS exam style (e.g. VCE/Further Maths,
IB, or senior applied maths tasks) where technology is essential to the method.
"""

TECH_ACTIVE_STYLE = """
Tech Active problems assume students have a CAS calculator (e.g. TI-Nspire CAS, Casio ClassPad).
These are extended-response tasks, not short contest problems.

Design principles:
- Require genuine use of CAS features: solve, factor, graph, numerical solve, tables, regression,
  or parameter exploration—not just arithmetic a calculator could do in one line.
- Do NOT reuse contest-style questions that become trivial with technology (e.g. bare factorisation,
  routine equation solving, or insight puzzles that CAS solves instantly without interpretation).
- Favour multi-part extended response: model a situation, use CAS to analyse, then interpret,
  justify, or communicate findings in context.
- Include interpretation, comparison, or verification steps after the calculator work.
- Problems may ask students to show calculator syntax used (e.g. "Using your CAS, ..." or
  "State the CAS command or menu path you would use, then ...").
- Use subparts (a), (b), (c) to scaffold modelling → calculator work → interpretation.
- Answers may be exact or decimal as appropriate, but reasoning and communication matter.
- Each problem should take noticeably longer than a contest question (roughly 8-15 minutes).
"""

CONTEST_STYLE = """
Contest-style challenge problems should feel like original problems from math competitions:
- Difficulty target: {contest_level} (late problems for that contest, not early warm-ups).
- Reward insight over computation: a clever substitution, symmetry, factorisation, or invariant
  should unlock the problem.
- Prefer exact answers (surds, simplified radicals, integers, or expressions in closed form).
- Use precise wording: "Find all...", "Determine the exact value of...", "Prove that...",
  "For how many integers...", "What is the minimum possible value of...".
- Avoid routine drill exercises in the challenge section (no bare "simplify sqrt(125)").
- Problems may combine ideas from the topic with contest staples (algebraic manipulation,
  number sense, inequalities, geometry setup, or counting) when natural.
- Each problem should be solvable in contest time (roughly 3-6 minutes) once the key idea is found.
- No calculator assumed.
"""

LATEX_RULES = """
LaTeX Formatting Rules:
1. Use the 'amsmath' and 'amsfonts' packages.
2. Ensure all math is wrapped in $...$ for inline and $$...$$ for display.
3. Output the RAW content only—no markdown wrappers (no ```latex fences).
4. Number questions from 1 and mark subparts with letters (a), (b), (c), etc.
5. Keep mathematical expressions on the same line as the question text unless a
   display equation is genuinely needed.
"""

DIAGRAM_STYLE = """
DIAGRAM MODE IS ON — work slowly and prioritise diagram accuracy over speed.

Use TikZ for all required figures. Before each diagram, write a one-line plan comment:
% Diagram plan: <shapes, given lengths/angles, labels>

TikZ setup (include in the document preamble):
\\usepackage{{tikz}}
\\usetikzlibrary{{angles, quotes, calc, positioning, arrows.meta}}

Drawing rules:
1. Plan vertex coordinates on paper first; use explicit \\coordinate (A) at (x,y);
2. Wrap each figure in \\begin{{center}}\\begin{{tikzpicture}}[scale=0.85] ... \\end{{tikzpicture}}\\end{{center}}
3. Label vertices clearly, placed away from edges (e.g. above left of A).
4. Mark right angles with the angles library; use tick marks for equal lengths.
5. Keep figures simple: one main shape per diagram, at most one auxiliary line.
6. The diagram must match the question — if AB = 5 cm is stated, the drawing must look consistent.
7. Avoid 3D, perspective, or heavily overlapping lines.

Question coverage:
- Include at least {diagrams_per_subtopic} diagram-backed question(s) per subtopic in Section A.
- Any question referencing a shape, angle, circle, or labelled point MUST include a diagram.
- Challenge problems may include diagrams when the visual setup is essential to the insight.

Check each diagram for: correct labels, readable scale, no missing points, compiles without TikZ errors.
"""

LATEX_RULES_DIAGRAM = LATEX_RULES + """
6. Include \\usepackage{tikz} and \\usetikzlibrary{angles, quotes, calc, positioning, arrows.meta}.
"""


def latex_rules(diagram_mode: bool = DIAGRAM_MODE) -> str:
    return LATEX_RULES_DIAGRAM if diagram_mode else LATEX_RULES


def diagram_instructions(
    diagram_mode: bool = DIAGRAM_MODE,
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
) -> str:
    if not diagram_mode:
        return ""
    return DIAGRAM_STYLE.format(diagrams_per_subtopic=diagrams_per_subtopic)


def setup_logging(log_file: Path = LOG_FILE) -> None:
    if logger.handlers:
        return

    log_file.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.setLevel(logging.DEBUG)
    logger.addHandler(console)
    logger.addHandler(file_handler)


@contextmanager
def log_step(step_name: str):
    logger.info("STEP START: %s", step_name)
    start = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - start
        logger.exception("STEP FAILED: %s (%.1fs)", step_name, elapsed)
        raise
    else:
        elapsed = time.perf_counter() - start
        logger.info("STEP DONE: %s (%.1fs)", step_name, elapsed)


def log_run_config(
    topic: str,
    subtopics: list[str],
    step: str,
    output_dir: Path,
    core_per_subtopic: int,
    challenge_per_subtopic: int,
    tech_active_per_subtopic: int,
    contest_level: str,
    diagram_mode: bool,
    diagrams_per_subtopic: int,
) -> None:
    cleaned = clean_subtopics(subtopics)
    paths = topic_paths(topic, output_dir)
    logger.info("=" * 60)
    logger.info("NEW RUN")
    logger.info("  Model: %s", MODEL)
    logger.info("  Step: %s", step)
    logger.info("  Topic: %s", topic)
    logger.info("  Subtopics (%d): %s", len(cleaned), cleaned)
    logger.info(
        "  Questions per subtopic: core=%d, challenge=%d, tech_active=%d",
        core_per_subtopic,
        challenge_per_subtopic,
        tech_active_per_subtopic,
    )
    logger.info(
        "  Totals: core=%d, challenge=%d, tech_active=%d",
        total_questions(cleaned, core_per_subtopic),
        total_questions(cleaned, challenge_per_subtopic),
        total_questions(cleaned, tech_active_per_subtopic),
    )
    logger.info("  Contest level: %s", contest_level)
    logger.info("  Diagram mode: %s", diagram_mode)
    if diagram_mode:
        logger.info("  Diagrams per subtopic: %d", diagrams_per_subtopic)
    logger.info("  Output dir: %s", output_dir)
    logger.info("  Research file: %s", paths["research"])
    logger.info("  Worksheet file: %s", paths["worksheet"])
    logger.info("  Solutions file: %s", paths["solutions"])
    logger.info("  Log file: %s", LOG_FILE)
    logger.info("  Batch API: %s", USE_BATCH_API)
    if USE_BATCH_API:
        logger.info("  Batch poll interval: %ds", BATCH_POLL_INTERVAL_SEC)
    logger.info("=" * 60)


def batch_config_dict(config: types.GenerateContentConfig | None) -> dict:
    if config is None:
        return {}

    batch_config: dict = {}
    if config.temperature is not None:
        batch_config["temperature"] = config.temperature
    if config.response_mime_type is not None:
        batch_config["response_mime_type"] = config.response_mime_type
    if config.tools:
        batch_config["tools"] = [{"google_search": {}}]
    return batch_config


def build_batch_request(prompt: str, **kwargs) -> dict:
    request = {
        "contents": [{"parts": [{"text": prompt}], "role": "user"}],
    }
    batch_config = batch_config_dict(kwargs.get("config"))
    if batch_config:
        request["config"] = batch_config
    return request


def extract_batch_response_text(inline_response) -> str:
    if inline_response.error:
        raise RuntimeError(f"Batch request failed: {inline_response.error}")

    response = inline_response.response
    if response is None:
        raise RuntimeError("Batch request returned no response payload.")

    text = response.text
    if not text:
        raise RuntimeError("Batch response was empty.")
    return text


def call_model_batch(client, call_name: str, prompt: str, **kwargs) -> str:
    model = kwargs.get("model", MODEL)
    request = build_batch_request(prompt, **kwargs)
    logger.info("Submitting batch job: %s (model=%s)", call_name, model)
    start = time.perf_counter()

    batch_job = client.batches.create(
        model=model,
        src=[request],
        config={"display_name": f"hw-{slugify(call_name)}"},
    )
    job_name = batch_job.name
    logger.info("Batch job created: %s", job_name)

    while True:
        batch_job = client.batches.get(name=job_name)
        state = batch_job.state.name
        if state in BATCH_TERMINAL_STATES:
            break
        logger.info("Batch job %s waiting... state=%s", call_name, state)
        time.sleep(BATCH_POLL_INTERVAL_SEC)

    elapsed = time.perf_counter() - start
    if state != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(f"Batch job {call_name} finished with state {state}")

    inline_responses = batch_job.dest.inlined_responses
    if not inline_responses:
        raise RuntimeError(f"Batch job {call_name} succeeded but returned no responses.")

    response_text = extract_batch_response_text(inline_responses[0])
    log_api_call(
        call_name,
        prompt,
        response_text,
        elapsed,
        model=model,
        mode="batch",
    )
    return response_text


def log_api_call(call_name: str, prompt: str, response_text: str, elapsed: float, **kwargs) -> None:
    details = ", ".join(f"{key}={value}" for key, value in kwargs.items())
    logger.info(
        "API CALL: %s | %.1fs | prompt=%d chars | response=%d chars%s",
        call_name,
        elapsed,
        len(prompt),
        len(response_text),
        f" | {details}" if details else "",
    )
    logger.debug("%s prompt preview: %s", call_name, prompt[:500])


def call_model(client, call_name: str, prompt: str, **kwargs) -> str:
    if USE_BATCH_API:
        return call_model_batch(client, call_name, prompt, **kwargs)

    model = kwargs.get("model", MODEL)
    logger.debug("Dispatching %s to model %s (realtime)", call_name, model)
    start = time.perf_counter()
    response = client.models.generate_content(**kwargs)
    elapsed = time.perf_counter() - start
    response_text = response.text
    log_api_call(call_name, prompt, response_text, elapsed, model=model, mode="realtime")
    return response_text


def slugify(text: str) -> str:
    slug = text.lower().strip()
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_") or "worksheet"


def clean_subtopics(subtopics: list[str]) -> list[str]:
    return [subtopic.strip() for subtopic in subtopics if subtopic.strip()]


def format_subtopics(subtopics: list[str]) -> str:
    return "\n".join(f"- {subtopic}" for subtopic in clean_subtopics(subtopics))


def total_questions(subtopics: list[str], per_subtopic: int) -> int:
    return len(clean_subtopics(subtopics)) * per_subtopic


def format_distribution(
    subtopics: list[str],
    core_per_subtopic: int,
    challenge_per_subtopic: int,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
) -> str:
    cleaned = clean_subtopics(subtopics)
    lines = ["Question distribution (strict — follow exactly):"]
    for subtopic in cleaned:
        lines.append(
            f"- {subtopic}: {core_per_subtopic} core question(s), "
            f"{challenge_per_subtopic} challenge problem(s), "
            f"{tech_active_per_subtopic} tech active question(s)"
        )
    lines.append(
        f"Total: {total_questions(cleaned, core_per_subtopic)} core questions, "
        f"{total_questions(cleaned, challenge_per_subtopic)} challenge problems, "
        f"{total_questions(cleaned, tech_active_per_subtopic)} tech active questions."
    )
    lines.append(
        "Do not assign extra questions to broader or harder subtopics. "
        "Each subtopic must receive exactly its quota."
    )
    return "\n".join(lines)


def research_topic(
    client,
    topic: str,
    subtopics: list[str],
    core_per_subtopic: int = CORE_QUESTIONS_PER_SUBTOPIC,
    challenge_per_subtopic: int = CHALLENGE_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level: str = CONTEST_LEVEL,
    diagram_mode: bool = DIAGRAM_MODE,
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
) -> str:
    cleaned = clean_subtopics(subtopics)
    subtopic_list = format_subtopics(cleaned)
    distribution = format_distribution(
        cleaned, core_per_subtopic, challenge_per_subtopic, tech_active_per_subtopic
    )
    contest_style = CONTEST_STYLE.format(contest_level=contest_level)
    diagram_block = diagram_instructions(diagram_mode, diagrams_per_subtopic)
    diagram_research = ""
    if diagram_mode:
        diagram_research = f"""
8. For each subtopic, plan at least {diagrams_per_subtopic} diagram-backed question(s).
   For each, sketch a plain-text diagram plan: vertex labels, coordinates or relative layout,
   given measurements, and what the student must find. Note common TikZ pitfalls to avoid.
"""
    prompt = f"""You are an experienced maths curriculum designer and contest problem writer.

Research the topic below using reputable online educational resources.
{REPUTABLE_SOURCES}

Topic: {topic}

Subtopics to cover:
{subtopic_list}

{distribution}
{diagram_block}

Produce structured research notes that include:
1. Key definitions, formulae, and techniques for each subtopic.
2. Typical question styles and difficulty progression (introductory to challenging).
3. Common misconceptions students make.
4. For each subtopic separately, exactly {core_per_subtopic} core practice question ideas.
   Do not give one subtopic more ideas because it is broader or harder.
5. For each subtopic separately, exactly {challenge_per_subtopic} contest-style challenge ideas.
{contest_style}
   For each challenge idea, note: the problem statement sketch, the key insight a strong
   student must spot, and why it fits {contest_level} difficulty.
6. For each subtopic separately, exactly {tech_active_per_subtopic} Tech Active (CAS calculator)
   extended-response question ideas.
{TECH_ACTIVE_STYLE}
   For each tech active idea, note: the problem sketch, which CAS features are genuinely needed,
   and why the task is NOT trivialised by having a calculator.
7. A short list of the online sources you consulted (name and URL where available).
{diagram_research}

Write clear plain-text notes only. Do not output LaTeX yet.
"""

    response_text = call_model(
        client,
        "research",
        prompt,
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.3 if diagram_mode else 0.4,
        ),
    )
    return response_text


def generate_questions_worksheet(
    client,
    topic: str,
    subtopics: list[str],
    research_notes: str,
    core_per_subtopic: int = CORE_QUESTIONS_PER_SUBTOPIC,
    challenge_per_subtopic: int = CHALLENGE_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level: str = CONTEST_LEVEL,
    diagram_mode: bool = DIAGRAM_MODE,
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
) -> str:
    cleaned = clean_subtopics(subtopics)
    num_core = total_questions(cleaned, core_per_subtopic)
    num_challenge = total_questions(cleaned, challenge_per_subtopic)
    num_tech_active = total_questions(cleaned, tech_active_per_subtopic)
    distribution = format_distribution(
        cleaned, core_per_subtopic, challenge_per_subtopic, tech_active_per_subtopic
    )
    subtopic_list = format_subtopics(cleaned)
    contest_style = CONTEST_STYLE.format(contest_level=contest_level)
    diagram_block = diagram_instructions(diagram_mode, diagrams_per_subtopic)
    role = "LaTeX expert, geometry diagram specialist, and contest problem writer" if diagram_mode else "LaTeX expert and contest problem writer"
    preamble = f"You are a {role}. Generate a complete, ready-to-compile .tex file."
    instruction = f"""
Create a homework worksheet for the topic "{topic}" using the research notes below.

Subtopics:
{subtopic_list}

{distribution}
{diagram_block}

Requirements:
1. Write original questions only—do not include solutions or answers.
2. Structure the worksheet into three clearly separated sections:

   **Section A — Core Practice**
   - Use a LaTeX subsection for each subtopic (in the order listed above).
   - In each subsection, include exactly {core_per_subtopic} core questions for that
     subtopic only—no more, no fewer.
   - Number core questions consecutively from 1 to {num_core} across the whole section.
   - Do not let broader or harder subtopics take extra questions.

   **Section B — Challenge Problems (Contest Style)**
   - Add a LaTeX section titled "Challenge Problems (Contest Style)".
   - Use a LaTeX subsection for each subtopic (same order as Section A).
   - In each subsection, include exactly {challenge_per_subtopic} challenge problems
     for that subtopic only.
   - Number challenge problems consecutively from C1 to C{num_challenge} across the section.
   - Write these as original contest problems at {contest_level} difficulty.
{contest_style}
   - Each challenge problem must hinge on a non-obvious insight, not longer computation.
   - Standalone problems only (no multi-part scaffolding that gives away the method).
   - Use subparts (a), (b), (c) only for UKMT-style multi-stage problems where later
     parts build on earlier results.

   **Section C — Tech Active (CAS Calculator Required)**
   - Add a LaTeX section titled "Tech Active (CAS Calculator Required)".
   - State clearly at the start of the section: "A CAS calculator is required for this section."
   - Use a LaTeX subsection for each subtopic (same order as Section A).
   - In each subsection, include exactly {tech_active_per_subtopic} extended-response
     questions for that subtopic only.
   - Number tech active questions consecutively from T1 to T{num_tech_active} across the section.
{TECH_ACTIVE_STYLE}
   - These must be qualitatively different from Section B: do not copy or lightly adapt
     contest problems that CAS would solve without meaningful interpretation.
   - Prefer multi-part questions with modelling, calculator use, and written interpretation.

3. Add a worksheet title matching the topic.
4. End with a short "Sources consulted" section listing the reputable resources
   referenced during question design (bullet list, plain text inside the document).

{latex_rules(diagram_mode)}
"""
    prompt = AI.generate_prompt(preamble=preamble, instruction=instruction, data=research_notes)

    response_text = call_model(
        client,
        "questions_worksheet",
        prompt,
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="text/plain",
            temperature=0.15 if diagram_mode else 0.25,
        ),
    )
    return response_text


def generate_solutions_worksheet(
    client,
    topic: str,
    subtopics: list[str],
    questions_worksheet: str,
    core_per_subtopic: int = CORE_QUESTIONS_PER_SUBTOPIC,
    challenge_per_subtopic: int = CHALLENGE_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level: str = CONTEST_LEVEL,
    diagram_mode: bool = DIAGRAM_MODE,
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
) -> str:
    cleaned = clean_subtopics(subtopics)
    num_core = total_questions(cleaned, core_per_subtopic)
    num_challenge = total_questions(cleaned, challenge_per_subtopic)
    num_tech_active = total_questions(cleaned, tech_active_per_subtopic)
    diagram_block = diagram_instructions(diagram_mode, diagrams_per_subtopic)
    diagram_solutions = ""
    if diagram_mode:
        diagram_solutions = """
10. For questions with diagrams, reproduce the original diagram verbatim, then add a second
    "Solution diagram" in grey/dashed lines if auxiliary constructions (altitudes, radii,
    angle bisectors) clarify the working.
"""
    preamble = "You are a LaTeX expert and contest solutions writer. Generate a complete, ready-to-compile .tex file."
    instruction = f"""
Create a worked-solutions sheet for the topic "{topic}" by solving the questions
in the worksheet below. Do not invent, reword, or substitute different questions.
{diagram_block}

Requirements:
1. Mirror the worksheet structure exactly, including each subtopic subsection:
   - Section A: solutions for core questions 1 to {num_core}
     ({core_per_subtopic} per subtopic)
   - Section B: solutions for contest-style challenge problems C1 to C{num_challenge}
     ({challenge_per_subtopic} per subtopic)
   - Section C: solutions for Tech Active questions T1 to T{num_tech_active}
     ({tech_active_per_subtopic} per subtopic)
2. For every question in the worksheet, reproduce the question text verbatim, then
   write the worked solution immediately after it.
3. Solve only the questions that appear in the worksheet—same numbering, same wording.
4. For contest-style challenge problems ({contest_level} level):
   - Begin each solution with a one-line "Key idea:" stating the insight.
   - Then give a clean solution path a contest student would write under time pressure.
   - Omit tedious arithmetic steps once the method is clear.
5. For Tech Active (CAS) questions:
   - Begin each solution with a one-line "CAS approach:" describing the calculator method.
   - Show representative CAS syntax or menu steps (e.g. solve(), graph + intersection,
     numerical solve, regression).
   - Include interpretation of calculator output, not just the final answer.
   - Note where a by-hand check or restriction on the domain is needed.
6. If a solution required inference beyond the question statement, label it
   "(AI-generated solution)" after that solution.
7. Add a title such as "{topic} — Worked Solutions".
8. Keep core solutions minimal but clear.
9. Do not include the "Sources consulted" section from the worksheet.
{diagram_solutions}

{latex_rules(diagram_mode)}
"""
    prompt = AI.generate_prompt(
        preamble=preamble,
        instruction=instruction,
        data=questions_worksheet,
    )

    response_text = call_model(
        client,
        "solutions_worksheet",
        prompt,
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="text/plain",
            temperature=0.1 if diagram_mode else 0.15,
        ),
    )
    return response_text


def topic_paths(topic: str, output_dir: Path = OUTPUT_DIR) -> dict[str, Path]:
    slug = slugify(topic)
    return {
        "research": output_dir / f"{slug}_research.txt",
        "worksheet": output_dir / f"{slug}_worksheet.tex",
        "solutions": output_dir / f"{slug}_solutions.tex",
    }


def read_step_input(path: Path, step_name: str) -> str:
    if not path.exists():
        logger.error("Missing input for %s: %s", step_name, path)
        raise FileNotFoundError(
            f"Missing {step_name} file: {path}\n"
            f"Run the previous step first (see STEP in the script)."
        )
    content = path.read_text(encoding="utf-8")
    logger.info("Read %s (%d chars) from %s", step_name, len(content), path)
    return content


def write_output(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    logger.info("Wrote %s (%d chars) to %s", path.name, len(content), path)


def format_python_query(topic: str, subtopics: list[str], timestamp: str) -> str:
    subtopic_lines = ",\n".join(f'    {subtopic!r}' for subtopic in subtopics)
    return (
        f"# {timestamp}\n"
        f"TOPIC = {topic!r}\n"
        f"SUBTOPICS = [\n{subtopic_lines}\n]\n"
    )


def log_past_query(
    topic: str,
    subtopics: list[str],
    path: Path = PAST_QUERIES_FILE,
) -> None:
    cleaned = clean_subtopics(subtopics)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n{format_python_query(topic, cleaned, timestamp)}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry)
    logger.info("Appended past query to %s", path)


def run_research_step(
    topic: str,
    subtopics: list[str],
    output_dir: Path = OUTPUT_DIR,
    core_per_subtopic: int = CORE_QUESTIONS_PER_SUBTOPIC,
    challenge_per_subtopic: int = CHALLENGE_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level: str = CONTEST_LEVEL,
    diagram_mode: bool = DIAGRAM_MODE,
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
) -> Path:
    cleaned = clean_subtopics(subtopics)
    if not cleaned:
        raise ValueError("At least one subtopic is required.")

    log_past_query(topic, cleaned)
    paths = topic_paths(topic, output_dir)

    with log_step("research"):
        client = AI.setup_client()
        logger.info("Gemini client initialised")
        research_notes = research_topic(
            client,
            topic,
            cleaned,
            core_per_subtopic,
            challenge_per_subtopic,
            tech_active_per_subtopic,
            contest_level,
            diagram_mode,
            diagrams_per_subtopic,
        )
        write_output(paths["research"], research_notes)
    logger.info('Next: set STEP = "questions" and re-run to generate the worksheet.')
    return paths["research"]


def run_questions_step(
    topic: str,
    subtopics: list[str],
    output_dir: Path = OUTPUT_DIR,
    core_per_subtopic: int = CORE_QUESTIONS_PER_SUBTOPIC,
    challenge_per_subtopic: int = CHALLENGE_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level: str = CONTEST_LEVEL,
    diagram_mode: bool = DIAGRAM_MODE,
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
) -> Path:
    cleaned = clean_subtopics(subtopics)
    if not cleaned:
        raise ValueError("At least one subtopic is required.")

    paths = topic_paths(topic, output_dir)
    research_notes = read_step_input(paths["research"], "research")

    with log_step("questions"):
        client = AI.setup_client()
        logger.info("Gemini client initialised")
        questions_tex = generate_questions_worksheet(
            client,
            topic,
            cleaned,
            research_notes,
            core_per_subtopic,
            challenge_per_subtopic,
            tech_active_per_subtopic,
            contest_level,
            diagram_mode,
            diagrams_per_subtopic,
        )
        write_output(paths["worksheet"], questions_tex)
    logger.info('Next: set STEP = "solutions" and re-run to generate worked solutions.')
    return paths["worksheet"]


def run_solutions_step(
    topic: str,
    subtopics: list[str],
    output_dir: Path = OUTPUT_DIR,
    core_per_subtopic: int = CORE_QUESTIONS_PER_SUBTOPIC,
    challenge_per_subtopic: int = CHALLENGE_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level: str = CONTEST_LEVEL,
    diagram_mode: bool = DIAGRAM_MODE,
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
) -> Path:
    cleaned = clean_subtopics(subtopics)
    if not cleaned:
        raise ValueError("At least one subtopic is required.")

    paths = topic_paths(topic, output_dir)
    questions_tex = read_step_input(paths["worksheet"], "worksheet")

    with log_step("solutions"):
        client = AI.setup_client()
        logger.info("Gemini client initialised")
        solutions_tex = generate_solutions_worksheet(
            client,
            topic,
            cleaned,
            questions_tex,
            core_per_subtopic,
            challenge_per_subtopic,
            tech_active_per_subtopic,
            contest_level,
            diagram_mode,
            diagrams_per_subtopic,
        )
        write_output(paths["solutions"], solutions_tex)
    return paths["solutions"]


def generate_worksheets(
    topic: str,
    subtopics: list[str],
    output_dir: Path = OUTPUT_DIR,
    core_per_subtopic: int = CORE_QUESTIONS_PER_SUBTOPIC,
    challenge_per_subtopic: int = CHALLENGE_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level: str = CONTEST_LEVEL,
    diagram_mode: bool = DIAGRAM_MODE,
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
    step: str = STEP,
) -> dict[str, Path]:
    setup_logging()
    log_run_config(
        topic,
        subtopics,
        step,
        output_dir,
        core_per_subtopic,
        challenge_per_subtopic,
        tech_active_per_subtopic,
        contest_level,
        diagram_mode,
        diagrams_per_subtopic,
    )
    if diagram_mode:
        logger.warning("Diagram mode ON — TikZ figures requested; verify PDFs carefully.")
    if USE_BATCH_API:
        logger.warning(
            "Batch API ON — 50%% token discount, but each call is async and may take minutes."
        )

    run_start = time.perf_counter()
    steps = {
        "research": lambda: {"research": run_research_step(
            topic,
            subtopics,
            output_dir,
            core_per_subtopic,
            challenge_per_subtopic,
            tech_active_per_subtopic,
            contest_level,
            diagram_mode,
            diagrams_per_subtopic,
        )},
        "questions": lambda: {"worksheet": run_questions_step(
            topic,
            subtopics,
            output_dir,
            core_per_subtopic,
            challenge_per_subtopic,
            tech_active_per_subtopic,
            contest_level,
            diagram_mode,
            diagrams_per_subtopic,
        )},
        "solutions": lambda: {"solutions": run_solutions_step(
            topic,
            subtopics,
            output_dir,
            core_per_subtopic,
            challenge_per_subtopic,
            tech_active_per_subtopic,
            contest_level,
            diagram_mode,
            diagrams_per_subtopic,
        )},
        "all": lambda: {
            "research": run_research_step(
                topic,
                subtopics,
                output_dir,
                core_per_subtopic,
                challenge_per_subtopic,
                tech_active_per_subtopic,
                contest_level,
                diagram_mode,
                diagrams_per_subtopic,
            ),
            "worksheet": run_questions_step(
                topic,
                subtopics,
                output_dir,
                core_per_subtopic,
                challenge_per_subtopic,
                tech_active_per_subtopic,
                contest_level,
                diagram_mode,
                diagrams_per_subtopic,
            ),
            "solutions": run_solutions_step(
                topic,
                subtopics,
                output_dir,
                core_per_subtopic,
                challenge_per_subtopic,
                tech_active_per_subtopic,
                contest_level,
                diagram_mode,
                diagrams_per_subtopic,
            ),
        },
    }

    if step not in steps:
        logger.error('Invalid STEP "%s"', step)
        raise ValueError(
            f'Invalid STEP "{step}". Choose from: research, questions, solutions, all'
        )

    results = steps[step]()
    elapsed = time.perf_counter() - run_start
    logger.info("RUN COMPLETE in %.1fs", elapsed)
    for name, path in results.items():
        logger.info("Output (%s): %s", name, path)
    return results


if __name__ == "__main__":
    try:
        results = generate_worksheets(
            topic=TOPIC,
            subtopics=SUBTOPICS,
            output_dir=OUTPUT_DIR,
            step=STEP,
        )
    except Exception:
        logger.error("Run aborted due to error.")
        raise
