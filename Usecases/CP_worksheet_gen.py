"""
Generate competitive programming worksheets from a topic and subtopics by researching
reputable CP resources, then producing a LaTeX problem sheet and matching worked solutions.

LEGACY VERSION: single-call solutions with automatic continuation if output truncates.

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
MODEL = "gemini-3-flash-preview"
OUTPUT_DIR = PROJECT_ROOT / "AI_output"
PAST_QUERIES_FILE = PROJECT_ROOT / "past queries.txt"
LOG_FILE = OUTPUT_DIR / "cp_worksheet_gen.log"

logger = logging.getLogger("cp_worksheet")

# --- Edit these before running ---
TOPIC = "Number theory"
SUBTOPICS = [
    "Finding factors",
    "Finding primes ",
    "Binary exponentiation",
    "Mathematical/bitwise construction",
    "Graph/Tree construction",
    "Modular inverse",
    "GCD",
    "Lowest common multiples",
    "Chinese remainder theorem",
    "Mobius inversion"
]

CONCEPTUAL_QUESTIONS_PER_SUBTOPIC = 3
CONTEST_EASY_QUESTIONS_PER_SUBTOPIC = 1
CONTEST_MEDIUM_QUESTIONS_PER_SUBTOPIC = 1
CONTEST_HARD_QUESTIONS_PER_SUBTOPIC = 1
TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC = 0
# Target difficulty for each contest section — edit freely, e.g. AtCoder ABC C, Codeforces Div. 2 B
CONTEST_LEVEL_EASY = "AtCoder ABC C"
CONTEST_LEVEL_MEDIUM = "AtCoder ABC D"
CONTEST_LEVEL_HARD = "AtCoder ABC E"
# Run one step at a time: "research", "questions", "solutions", or "all"
STEP = "all"
# Set True for geometry/visual topics (uses TikZ diagrams in LaTeX output)
DIAGRAM_MODE = False
DIAGRAMS_PER_SUBTOPIC = 1  # minimum diagram-backed questions per subtopic when DIAGRAM_MODE is True
# Use Gemini Batch API for 50% token discount (async — may take minutes per call)
USE_BATCH_API = False
BATCH_POLL_INTERVAL_SEC = 30
MAX_OUTPUT_TOKENS = 81920
SOLUTIONS_CONTINUATION_MAX = 8
SOLUTIONS_MIN_CHARS_RATIO = 0.4

BATCH_TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}

REPUTABLE_SOURCES = """
Prioritise content from reputable competitive programming sources such as:
- CP-Algorithms, USACO Guide, CSES Problem Set, Codeforces EDU / Catalog
- AtCoder Beginner/Regular Contest archives (for style only — write original problems)
- ICPC problem archives, Kattis, and official contest editorials for technique reference
- William Lin / Errichto / SecondThread educational material, USACO training pages
For contest problems, study the style of Codeforces Div. 2 A–D, AtCoder ABC B–F,
USACO Bronze/Silver/Gold, and ICPC regional problems — but write original problems;
never copy existing contest statements.
For implementation/analysis problems, draw on tasks where careful I/O handling,
edge cases, complexity justification, and pseudocode or solution sketches matter.
"""

IMPLEMENTATION_STYLE = """
Implementation & Analysis problems assume students are training for timed CP contests
(C++ or Python). These are extended-response tasks, not bare "implement this loop" drills.

Design principles:
- Require genuine algorithmic design: state the approach, justify correctness briefly,
  and analyse time/space complexity (Big-O with n, m, etc. as in the constraints).
- Include realistic constraints (e.g. 1 ≤ n ≤ 2×10^5, time limit 2s) when appropriate.
- Use standard CP problem format when helpful: Input / Output / Constraints / Example.
- Favour multi-part extended response: (a) explain the idea, (b) give pseudocode or
  key implementation details, (c) discuss edge cases or why the complexity fits.
- Do NOT ask for full compilable source code in the worksheet — focus on algorithm,
  complexity, and implementation pitfalls (overflow, 1-indexing, recursion depth, etc.).
- Problems may ask students to trace the algorithm on a small example or identify
  why a naive approach TLEs.
- Each problem should take noticeably longer than a single contest problem
  (roughly 10–20 minutes including written analysis).
"""

CONCEPTUAL_STYLE = """
Conceptual questions test understanding of ideas, not full contest implementation:
- Ask "explain why", "what is wrong with this approach", "trace on a small example",
  "compare two strategies", or "identify the key invariant".
- Prefer short, precise prompts answerable in a few sentences or a small hand trace.
- May include tiny numeric examples, pseudocode snippets to critique, or true/false with justification.
- Do NOT use full Input/Output/Constraints contest statements unless a miniature example helps.
- These should build intuition for the subtopic before the contest sections.
"""

CONTEST_STYLE = """
Contest-style challenge problems should feel like original problems from CP contests:
- Difficulty target: {contest_level} (solid for that level, not trivial implementation).
- Reward the right algorithm/data structure choice: a non-obvious greedy, DP state,
  graph reduction, binary search on answer, or invariant — not brute force with tweaks.
- Include clear Input/Output/Constraints when the problem is implementation-oriented;
  for pure math/constructive CP problems, use precise "Find...", "Determine...", "Prove...".
- Prefer exact integer answers, YES/NO, or minimum/maximum values with justification.
- Avoid school-math drill (no bare "simplify" or unrelated calculus).
- Problems may combine the topic with CP staples (two pointers, sorting, prefix sums,
  mod arithmetic, bit tricks) when natural.
- Each problem should be solvable in contest time (roughly 15–45 minutes at the target level)
  once the key idea is found.
- No calculator assumed; complexity must fit typical limits (O(n log n), O(n), etc.).
"""

LATEX_RULES = """
LaTeX Formatting Rules:
1. Use the 'amsmath' and 'amsfonts' packages; add \\usepackage[hidelinks]{hyperref} for URLs.
2. Ensure all math is wrapped in $...$ for inline and $$...$$ for display.
3. Output the RAW content only—no markdown wrappers (no ```latex fences).
4. Number questions from 1 and mark subparts with letters (a), (b), (c), etc.
5. Keep mathematical expressions on the same line as the problem text unless a
   display equation is genuinely needed.
6. For pseudocode or sample I/O, use \\begin{verbatim}...\\end{verbatim} or \\texttt{} for inline.
7. For Input/Output/Constraints blocks, use a compact format (e.g. \\textbf{Input:} then verbatim).
8. Do not use \\renewcommand{\\item} with hyperref; use enumitem [resume] if continuing lists.
9. Put URLs in \\url{...}, not \\texttt{...}.
"""

DIAGRAM_STYLE = """
DIAGRAM MODE IS ON — work slowly and prioritise diagram accuracy over speed.

Use TikZ for all required figures (graphs, trees, grids, flow sketches). Before each diagram,
write a one-line plan comment: % Diagram plan: <vertices/edges, labels, what is highlighted>

TikZ setup (include in the document preamble):
\\usepackage{{tikz}}
\\usetikzlibrary{{graphs, graphdrawing, positioning, arrows.meta, calc}}
\\usegdlibrary{{force, layered}}

Drawing rules:
1. For graphs/trees: use explicit node coordinates or a simple layered layout; label vertices clearly.
2. Wrap each figure in \\begin{{center}}\\begin{{tikzpicture}}[scale=0.85] ... \\end{{tikzpicture}}\\end{{center}}
3. Highlight BFS/DFS order, shortest paths, or DSU merges when the question depends on them.
4. Keep figures readable: avoid crossing edges where possible; use arrows for directed graphs.
5. The diagram must match the problem statement (edge weights, directions, node names).

Question coverage:
- Include at least {diagrams_per_subtopic} diagram-backed question(s) per subtopic in Section A (Conceptual).
- Any problem referencing a graph, tree, grid, or network MUST include a diagram when setup is non-trivial.
- Contest problems may include diagrams when the visual structure is essential to the insight.

Check each diagram for: correct labels, readable layout, compiles without TikZ errors.
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
    conceptual_per_subtopic: int,
    contest_easy_per_subtopic: int,
    contest_medium_per_subtopic: int,
    contest_hard_per_subtopic: int,
    tech_active_per_subtopic: int,
    contest_level_easy: str,
    contest_level_medium: str,
    contest_level_hard: str,
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
        "  Questions per subtopic: conceptual=%d, contest_easy=%d, contest_medium=%d, contest_hard=%d, tech_active=%d",
        conceptual_per_subtopic,
        contest_easy_per_subtopic,
        contest_medium_per_subtopic,
        contest_hard_per_subtopic,
        tech_active_per_subtopic,
    )
    logger.info(
        "  Totals: conceptual=%d, contest_easy=%d, contest_medium=%d, contest_hard=%d, tech_active=%d",
        total_questions(cleaned, conceptual_per_subtopic),
        total_questions(cleaned, contest_easy_per_subtopic),
        total_questions(cleaned, contest_medium_per_subtopic),
        total_questions(cleaned, contest_hard_per_subtopic),
        total_questions(cleaned, tech_active_per_subtopic),
    )
    logger.info("  Contest level (easy): %s", contest_level_easy)
    logger.info("  Contest level (medium): %s", contest_level_medium)
    logger.info("  Contest level (hard): %s", contest_level_hard)
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


class TruncatedResponseError(RuntimeError):
    """Raised when model output hits the token limit and continuation did not finish."""


def generation_config(**kwargs) -> types.GenerateContentConfig:
    if "max_output_tokens" not in kwargs:
        kwargs["max_output_tokens"] = MAX_OUTPUT_TOKENS
    return types.GenerateContentConfig(**kwargs)


def response_finish_reason(response) -> str | None:
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return None
    reason = candidates[0].finish_reason
    return str(reason) if reason is not None else None


def extract_response_text(response) -> str:
    text = getattr(response, "text", None)
    if text and text.strip():
        return text.strip()

    chunks: list[str] = []
    for candidate in getattr(response, "candidates", None) or []:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", None) or []:
            part_text = getattr(part, "text", None)
            if part_text:
                chunks.append(part_text)
    return "\n".join(chunks).strip()


def response_hit_token_limit(response) -> bool:
    reason = response_finish_reason(response)
    return bool(reason and "MAX_TOKENS" in reason.upper())


def strip_markdown_fences(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:latex|tex)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def strip_continuation_preamble(text: str) -> str:
    cleaned = strip_markdown_fences(text)
    if "\\begin{document}" in cleaned:
        cleaned = cleaned.split("\\begin{document}", 1)[1]
    cleaned = re.sub(
        r"\\documentclass[\s\S]*?\\begin\{document\}",
        "",
        cleaned,
        count=1,
    )
    cleaned = re.sub(r"\\maketitle\s*", "", cleaned, count=1)
    return cleaned.strip()


def join_solution_parts(parts: list[str]) -> str:
    if not parts:
        return ""

    merged = strip_markdown_fences(parts[0])
    for idx, part in enumerate(parts[1:], start=1):
        chunk = strip_continuation_preamble(part)
        if idx < len(parts) - 1:
            chunk = re.sub(r"\\end\{document\}\s*$", "", chunk).strip()
        if chunk:
            merged = merged.rstrip() + "\n\n" + chunk

    return merged.rstrip() + "\n"


def finalize_solution_document(text: str) -> str:
    cleaned = text.rstrip()
    if not cleaned.endswith("\\end{document}"):
        cleaned += "\n\\end{document}"
    return cleaned + "\n"


def solutions_document_complete(text: str, worksheet: str = "") -> bool:
    if not text.rstrip().endswith("\\end{document}"):
        return False
    if worksheet:
        min_chars = max(1000, int(len(worksheet) * SOLUTIONS_MIN_CHARS_RATIO))
        if len(text) < min_chars:
            return False
    for marker in ("Section C", "Section D", "Section E"):
        if marker not in text:
            return False
    return True


def validate_solutions_document(text: str, worksheet: str, expected_questions: int) -> None:
    min_chars = max(1000, int(len(worksheet) * SOLUTIONS_MIN_CHARS_RATIO))
    if len(text) < min_chars:
        raise ValueError(
            f"Solutions document too short ({len(text)} chars; expected at least {min_chars})."
        )
    if not solutions_document_complete(text, worksheet):
        raise ValueError("Solutions document is missing \\end{document}.")


def batch_config_dict(config: types.GenerateContentConfig | None) -> dict:
    if config is None:
        return {}

    batch_config: dict = {}
    if config.temperature is not None:
        batch_config["temperature"] = config.temperature
    if config.response_mime_type is not None:
        batch_config["response_mime_type"] = config.response_mime_type
    if config.max_output_tokens is not None:
        batch_config["max_output_tokens"] = config.max_output_tokens
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

    text = extract_response_text(response)
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
        config={"display_name": f"cp-{slugify(call_name)}"},
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


def invoke_model(client, call_name: str, prompt: str, **kwargs) -> tuple[str, object]:
    if USE_BATCH_API:
        text = call_model_batch(client, call_name, prompt, **kwargs)
        return text, None

    model = kwargs.get("model", MODEL)
    logger.debug("Dispatching %s to model %s (realtime)", call_name, model)
    start = time.perf_counter()
    response = client.models.generate_content(**kwargs)
    elapsed = time.perf_counter() - start
    response_text = extract_response_text(response)
    finish_reason = response_finish_reason(response)
    log_api_call(
        call_name,
        prompt,
        response_text,
        elapsed,
        model=model,
        mode="realtime",
        finish_reason=finish_reason,
    )
    return response_text, response


def call_model(client, call_name: str, prompt: str, **kwargs) -> str:
    response_text, _ = invoke_model(client, call_name, prompt, **kwargs)
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
    conceptual_per_subtopic: int,
    contest_easy_per_subtopic: int,
    contest_medium_per_subtopic: int,
    contest_hard_per_subtopic: int,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
) -> str:
    cleaned = clean_subtopics(subtopics)
    lines = ["Question distribution (strict — follow exactly):"]
    for subtopic in cleaned:
        lines.append(
            f"- {subtopic}: {conceptual_per_subtopic} conceptual question(s), "
            f"{contest_easy_per_subtopic} contest-easy problem(s), "
            f"{contest_medium_per_subtopic} contest-medium problem(s), "
            f"{contest_hard_per_subtopic} contest-hard problem(s), "
            f"{tech_active_per_subtopic} implementation & analysis question(s)"
        )
    lines.append(
        f"Total: {total_questions(cleaned, conceptual_per_subtopic)} conceptual questions, "
        f"{total_questions(cleaned, contest_easy_per_subtopic)} contest-easy problems, "
        f"{total_questions(cleaned, contest_medium_per_subtopic)} contest-medium problems, "
        f"{total_questions(cleaned, contest_hard_per_subtopic)} contest-hard problems, "
        f"{total_questions(cleaned, tech_active_per_subtopic)} implementation & analysis questions."
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
    conceptual_per_subtopic: int = CONCEPTUAL_QUESTIONS_PER_SUBTOPIC,
    contest_easy_per_subtopic: int = CONTEST_EASY_QUESTIONS_PER_SUBTOPIC,
    contest_medium_per_subtopic: int = CONTEST_MEDIUM_QUESTIONS_PER_SUBTOPIC,
    contest_hard_per_subtopic: int = CONTEST_HARD_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level_easy: str = CONTEST_LEVEL_EASY,
    contest_level_medium: str = CONTEST_LEVEL_MEDIUM,
    contest_level_hard: str = CONTEST_LEVEL_HARD,
    diagram_mode: bool = DIAGRAM_MODE,
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
) -> str:
    cleaned = clean_subtopics(subtopics)
    subtopic_list = format_subtopics(cleaned)
    distribution = format_distribution(
        cleaned,
        conceptual_per_subtopic,
        contest_easy_per_subtopic,
        contest_medium_per_subtopic,
        contest_hard_per_subtopic,
        tech_active_per_subtopic,
    )
    contest_style_easy = CONTEST_STYLE.format(contest_level=contest_level_easy)
    contest_style_medium = CONTEST_STYLE.format(contest_level=contest_level_medium)
    contest_style_hard = CONTEST_STYLE.format(contest_level=contest_level_hard)
    diagram_block = diagram_instructions(diagram_mode, diagrams_per_subtopic)
    diagram_research = ""
    if diagram_mode:
        diagram_research = f"""
9. For each subtopic, plan at least {diagrams_per_subtopic} diagram-backed question(s).
   For each, sketch a plain-text diagram plan: graph/tree nodes, edges, weights or directions,
   and what the student must compute or trace. Note common TikZ pitfalls to avoid.
"""
    prompt = f"""You are an experienced competitive programming coach and problem setter.

Research the topic below using reputable online CP resources.
{REPUTABLE_SOURCES}

Topic: {topic}

Subtopics to cover:
{subtopic_list}

{distribution}
{diagram_block}

Produce structured research notes that include:
1. Key definitions, algorithms, data structures, and techniques for each subtopic.
2. Typical problem patterns, constraints, and difficulty progression (warm-up to contest).
3. Common implementation pitfalls and wrong approaches (TLE, MLE, WA on edge cases).
4. For each subtopic separately, exactly {conceptual_per_subtopic} conceptual question ideas
   (explain why, compare approaches, trace small examples, critique pseudocode).
{CONCEPTUAL_STYLE}
   Do not give one subtopic more ideas because it is broader or harder.
5. For each subtopic separately, exactly {contest_easy_per_subtopic} contest-style problem ideas
   at {contest_level_easy} difficulty.
{contest_style_easy}
   For each easy-contest idea, note: problem sketch (with constraints if relevant), the key
   algorithmic insight, expected complexity, and why it fits {contest_level_easy} difficulty.
6. For each subtopic separately, exactly {contest_medium_per_subtopic} contest-style problem ideas
   at {contest_level_medium} difficulty.
{contest_style_medium}
   For each medium-contest idea, note: problem sketch (with constraints if relevant), the key
   algorithmic insight, expected complexity, and why it fits {contest_level_medium} difficulty.
7. For each subtopic separately, exactly {contest_hard_per_subtopic} contest-style problem ideas
   at {contest_level_hard} difficulty.
{contest_style_hard}
   For each hard-contest idea, note: problem sketch (with constraints if relevant), the key
   algorithmic insight, expected complexity, and why it fits {contest_level_hard} difficulty.
8. For each subtopic separately, exactly {tech_active_per_subtopic} Implementation & Analysis
   extended-response problem ideas.
{IMPLEMENTATION_STYLE}
   For each implementation idea, note: the problem sketch, what must be analysed in writing
   (correctness, complexity, edge cases), and why it is not a one-line implementation task.
9. A short list of the online sources you consulted (name and URL where available).
{diagram_research}

Write clear plain-text notes only. Do not output LaTeX yet.
"""

    response_text = call_model(
        client,
        "research",
        prompt,
        model=MODEL,
        contents=prompt,
        config=generation_config(
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
    conceptual_per_subtopic: int = CONCEPTUAL_QUESTIONS_PER_SUBTOPIC,
    contest_easy_per_subtopic: int = CONTEST_EASY_QUESTIONS_PER_SUBTOPIC,
    contest_medium_per_subtopic: int = CONTEST_MEDIUM_QUESTIONS_PER_SUBTOPIC,
    contest_hard_per_subtopic: int = CONTEST_HARD_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level_easy: str = CONTEST_LEVEL_EASY,
    contest_level_medium: str = CONTEST_LEVEL_MEDIUM,
    contest_level_hard: str = CONTEST_LEVEL_HARD,
    diagram_mode: bool = DIAGRAM_MODE,
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
) -> str:
    cleaned = clean_subtopics(subtopics)
    num_conceptual = total_questions(cleaned, conceptual_per_subtopic)
    num_contest_easy = total_questions(cleaned, contest_easy_per_subtopic)
    num_contest_medium = total_questions(cleaned, contest_medium_per_subtopic)
    num_contest_hard = total_questions(cleaned, contest_hard_per_subtopic)
    num_tech_active = total_questions(cleaned, tech_active_per_subtopic)
    distribution = format_distribution(
        cleaned,
        conceptual_per_subtopic,
        contest_easy_per_subtopic,
        contest_medium_per_subtopic,
        contest_hard_per_subtopic,
        tech_active_per_subtopic,
    )
    subtopic_list = format_subtopics(cleaned)
    contest_style_easy = CONTEST_STYLE.format(contest_level=contest_level_easy)
    contest_style_medium = CONTEST_STYLE.format(contest_level=contest_level_medium)
    contest_style_hard = CONTEST_STYLE.format(contest_level=contest_level_hard)
    diagram_block = diagram_instructions(diagram_mode, diagrams_per_subtopic)
    role = (
        "LaTeX expert, CP problem setter, and diagram specialist"
        if diagram_mode
        else "LaTeX expert and competitive programming problem setter"
    )
    preamble = f"You are a {role}. Generate a complete, ready-to-compile .tex file."
    instruction = f"""
Create a competitive programming worksheet for the topic "{topic}" using the research notes below.

Subtopics:
{subtopic_list}

{distribution}
{diagram_block}

Requirements:
1. Write original problems only—do not include solutions or answers.
2. Structure the worksheet into five clearly separated sections:

   **Section A — Conceptual Questions**
   - Use a LaTeX subsection for each subtopic (in the order listed above).
   - In each subsection, include exactly {conceptual_per_subtopic} conceptual questions for that
     subtopic only—no more, no fewer.
{CONCEPTUAL_STYLE}
   - Number conceptual questions consecutively from 1 to {num_conceptual} across the whole section.

   **Section B — Contest Problems (Easy)**
   - Add a LaTeX section titled "Contest Problems — Easy ({contest_level_easy})".
   - Use a LaTeX subsection for each subtopic (same order as Section A).
   - In each subsection, include exactly {contest_easy_per_subtopic} contest problems
     for that subtopic only.
   - Number easy contest problems consecutively from E1 to E{num_contest_easy} across the section.
   - Write these as original CP problems at {contest_level_easy} difficulty.
{contest_style_easy}
   - Use Input / Output / Constraints / Example blocks when the problem is implementation-style.
   - Each problem must hinge on choosing the right algorithm or data structure, not longer brute force.
   - Standalone problems only (no multi-part scaffolding that gives away the full solution path).

   **Section C — Contest Problems (Medium)**
   - Add a LaTeX section titled "Contest Problems — Medium ({contest_level_medium})".
   - Use a LaTeX subsection for each subtopic (same order as Section A).
   - In each subsection, include exactly {contest_medium_per_subtopic} contest problems
     for that subtopic only.
   - Number medium contest problems consecutively from M1 to M{num_contest_medium} across the section.
   - Write these as original CP problems at {contest_level_medium} difficulty.
{contest_style_medium}
   - Use Input / Output / Constraints / Example blocks when the problem is implementation-style.
   - Each problem must hinge on choosing the right algorithm or data structure, not longer brute force.
   - Standalone problems only (no multi-part scaffolding that gives away the full solution path).
   - These must be qualitatively harder than Section B — not just longer versions of the easy problems.

   **Section D — Contest Problems (Hard)**
   - Add a LaTeX section titled "Contest Problems — Hard ({contest_level_hard})".
   - Use a LaTeX subsection for each subtopic (same order as Section A).
   - In each subsection, include exactly {contest_hard_per_subtopic} contest problems
     for that subtopic only.
   - Number hard contest problems consecutively from H1 to H{num_contest_hard} across the section.
   - Write these as original CP problems at {contest_level_hard} difficulty.
{contest_style_hard}
   - Use Input / Output / Constraints / Example blocks when the problem is implementation-style.
   - Each problem must hinge on choosing the right algorithm or data structure, not longer brute force.
   - Standalone problems only (no multi-part scaffolding that gives away the full solution path).
   - These must be qualitatively harder than Section C — not just longer versions of the medium problems.

   **Section E — Implementation & Analysis**
   - Add a LaTeX section titled "Implementation \\& Analysis".
   - State clearly at the start: "Extended written responses required; focus on algorithm,
     complexity, and edge cases (no full source code required on the worksheet)."
   - Use a LaTeX subsection for each subtopic (same order as Section A).
   - In each subsection, include exactly {tech_active_per_subtopic} extended-response
     problems for that subtopic only.
   - Number these consecutively from I1 to I{num_tech_active} across the section.
{IMPLEMENTATION_STYLE}
   - These must be qualitatively different from Sections B, C, and D: emphasise analysis, pitfalls,
     and justification—not just "submit a solution to an online judge".
   - Prefer multi-part questions with (a) algorithm outline, (b) complexity, (c) edge cases.

3. Add a worksheet title matching the topic (e.g. "Graph Algorithms — CP Worksheet").
4. End with a short "Sources consulted" section listing CP resources referenced (bullet list).

{latex_rules(diagram_mode)}
"""
    prompt = AI.generate_prompt(preamble=preamble, instruction=instruction, data=research_notes)

    response_text = call_model(
        client,
        "questions_worksheet",
        prompt,
        model=MODEL,
        contents=prompt,
        config=generation_config(
            response_mime_type="text/plain",
            temperature=0.15 if diagram_mode else 0.25,
        ),
    )
    return response_text


def build_solutions_continuation_prompt(
    topic: str,
    num_conceptual: int,
    num_contest_easy: int,
    num_contest_medium: int,
    num_contest_hard: int,
    num_tech_active: int,
    tail: str,
) -> str:
    return f"""Continue the LaTeX worked-solutions document for "{topic}" EXACTLY where you stopped.

Rules:
1. Output ONLY the next raw LaTeX fragment — no markdown fences.
2. Do NOT repeat any content already written below.
3. Do NOT output \\documentclass, \\usepackage, or \\begin{{document}} again.
4. Continue question numbering from where you left off until ALL remaining questions are solved:
   - Section A conceptual questions through {num_conceptual}
   - Section B easy contest problems through E{num_contest_easy}
   - Section C medium contest problems through M{num_contest_medium}
   - Section D hard contest problems through H{num_contest_hard}
   - Section E implementation & analysis questions through I{num_tech_active}
5. End the full document with \\end{{document}} once every question is complete.

Last lines already written (continue immediately after this):
{tail}
"""


def generate_solutions_worksheet(
    client,
    topic: str,
    subtopics: list[str],
    questions_worksheet: str,
    conceptual_per_subtopic: int = CONCEPTUAL_QUESTIONS_PER_SUBTOPIC,
    contest_easy_per_subtopic: int = CONTEST_EASY_QUESTIONS_PER_SUBTOPIC,
    contest_medium_per_subtopic: int = CONTEST_MEDIUM_QUESTIONS_PER_SUBTOPIC,
    contest_hard_per_subtopic: int = CONTEST_HARD_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level_easy: str = CONTEST_LEVEL_EASY,
    contest_level_medium: str = CONTEST_LEVEL_MEDIUM,
    contest_level_hard: str = CONTEST_LEVEL_HARD,
    diagram_mode: bool = DIAGRAM_MODE,
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
) -> str:
    cleaned = clean_subtopics(subtopics)
    num_conceptual = total_questions(cleaned, conceptual_per_subtopic)
    num_contest_easy = total_questions(cleaned, contest_easy_per_subtopic)
    num_contest_medium = total_questions(cleaned, contest_medium_per_subtopic)
    num_contest_hard = total_questions(cleaned, contest_hard_per_subtopic)
    num_tech_active = total_questions(cleaned, tech_active_per_subtopic)
    diagram_block = diagram_instructions(diagram_mode, diagrams_per_subtopic)
    diagram_solutions = ""
    if diagram_mode:
        diagram_solutions = """
10. For questions with graph/tree diagrams, reproduce the original diagram verbatim, then add a
    second "Solution diagram" in grey/dashed lines if auxiliary edges, visited order, or path
    highlighting clarifies the working.
"""
    preamble = (
        "You are a LaTeX expert and competitive programming solutions writer. "
        "Generate a complete, ready-to-compile .tex file."
    )
    instruction = f"""
Create a worked-solutions sheet for the topic "{topic}" by solving the problems
in the worksheet below. Do not invent, reword, or substitute different problems.
{diagram_block}

Requirements:
1. Mirror the worksheet structure exactly, including each subtopic subsection:
   - Section A: solutions for conceptual questions 1 to {num_conceptual}
     ({conceptual_per_subtopic} per subtopic)
   - Section B: solutions for easy contest problems E1 to E{num_contest_easy}
     ({contest_easy_per_subtopic} per subtopic, {contest_level_easy} level)
   - Section C: solutions for medium contest problems M1 to M{num_contest_medium}
     ({contest_medium_per_subtopic} per subtopic, {contest_level_medium} level)
   - Section D: solutions for hard contest problems H1 to H{num_contest_hard}
     ({contest_hard_per_subtopic} per subtopic, {contest_level_hard} level)
   - Section E: solutions for Implementation & Analysis questions I1 to I{num_tech_active}
     ({tech_active_per_subtopic} per subtopic)
2. For every problem, give the question number/label from the worksheet followed by the
   worked solution. Repeat the full problem text only if it is short; otherwise use the
   label (e.g. \\item[12.], \\textbf{{E4.}}, \\textbf{{M3.}}, \\textbf{{H2.}}, \\textbf{{I2.}}) and \\textbf{{Solution:}}.
3. Solve only the problems that appear in the worksheet—same numbering, same meaning.
4. For easy contest problems ({contest_level_easy} level):
   - Begin each solution with a one-line "Key idea:" stating the algorithm or insight.
   - Then give a clean solution path: approach, correctness sketch, and complexity.
   - Include sample trace or small example when it clarifies the method.
   - Omit tedious step-by-step arithmetic once the method is clear.
5. For medium contest problems ({contest_level_medium} level):
   - Begin each solution with a one-line "Key idea:" stating the algorithm or insight.
   - Then give a clean solution path: approach, correctness sketch, and complexity.
   - Include sample trace or small example when it clarifies the method.
   - Omit tedious step-by-step arithmetic once the method is clear.
6. For hard contest problems ({contest_level_hard} level):
   - Begin each solution with a one-line "Key idea:" stating the algorithm or insight.
   - Then give a clean solution path: approach, correctness sketch, and complexity.
   - Include sample trace or small example when it clarifies the method.
   - Omit tedious step-by-step arithmetic once the method is clear.
7. For Implementation & Analysis questions:
   - Begin each solution with a one-line "Approach:" summarising the algorithm.
   - Give pseudocode or key implementation steps in \\begin{{verbatim}} when helpful.
   - State time and space complexity explicitly (Big-O with problem variables).
   - Discuss edge cases, overflow, indexing, and why naive approaches fail.
   - Do not dump full 200-line source code—focus on what a strong trainee would write in contest notes.
7. If a solution required inference beyond the problem statement, label it
   "(AI-generated solution)" after that solution.
8. Add a title such as "{topic} — Worked Solutions".
9. Keep conceptual solutions minimal but clear.
10. Do not include the "Sources consulted" section from the worksheet.
{diagram_solutions}

{latex_rules(diagram_mode)}
"""
    prompt = AI.generate_prompt(
        preamble=preamble,
        instruction=instruction,
        data=questions_worksheet,
    )
    expected_questions = (
        num_conceptual
        + num_contest_easy
        + num_contest_medium
        + num_contest_hard
        + num_tech_active
    )
    config = generation_config(
        response_mime_type="text/plain",
        temperature=0.1 if diagram_mode else 0.15,
    )
    request_kwargs = {
        "model": MODEL,
        "contents": prompt,
        "config": config,
    }

    parts: list[str] = []
    continuation_prompt = prompt
    truncated = False

    for part_index in range(SOLUTIONS_CONTINUATION_MAX + 1):
        call_name = (
            "solutions_worksheet"
            if part_index == 0
            else f"solutions_worksheet_cont_{part_index + 1}"
        )
        part_text, response = invoke_model(
            client,
            call_name,
            continuation_prompt,
            **request_kwargs,
        )
        if not part_text.strip():
            raise RuntimeError(f"{call_name} returned empty text.")

        parts.append(part_text)
        merged = join_solution_parts(parts)
        truncated = response is not None and response_hit_token_limit(response)
        complete = solutions_document_complete(merged, questions_worksheet) and not truncated

        logger.info(
            "Solutions part %d: %d chars (merged total %d chars, complete=%s, truncated=%s)",
            part_index + 1,
            len(part_text),
            len(merged),
            complete,
            truncated,
        )

        if complete:
            validate_solutions_document(merged, questions_worksheet, expected_questions)
            return merged

        if part_index >= SOLUTIONS_CONTINUATION_MAX:
            break

        tail = merged[-2000:]
        continuation_prompt = build_solutions_continuation_prompt(
            topic,
            num_conceptual,
            num_contest_easy,
            num_contest_medium,
            num_contest_hard,
            num_tech_active,
            tail,
        )
        request_kwargs["contents"] = continuation_prompt

    merged = join_solution_parts(parts)
    if solutions_document_complete(merged, questions_worksheet):
        validate_solutions_document(merged, questions_worksheet, expected_questions)
        return merged

    raise TruncatedResponseError(
        f"Solutions still incomplete after {len(parts)} part(s) "
        f"({len(merged)} chars). Last finish was truncated={truncated}. "
        "Try STEP='solutions' again to continue from the partial output."
    )


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
    conceptual_per_subtopic: int = CONCEPTUAL_QUESTIONS_PER_SUBTOPIC,
    contest_easy_per_subtopic: int = CONTEST_EASY_QUESTIONS_PER_SUBTOPIC,
    contest_medium_per_subtopic: int = CONTEST_MEDIUM_QUESTIONS_PER_SUBTOPIC,
    contest_hard_per_subtopic: int = CONTEST_HARD_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level_easy: str = CONTEST_LEVEL_EASY,
    contest_level_medium: str = CONTEST_LEVEL_MEDIUM,
    contest_level_hard: str = CONTEST_LEVEL_HARD,
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
            conceptual_per_subtopic,
            contest_easy_per_subtopic,
            contest_medium_per_subtopic,
            contest_hard_per_subtopic,
            tech_active_per_subtopic,
            contest_level_easy,
            contest_level_medium,
            contest_level_hard,
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
    conceptual_per_subtopic: int = CONCEPTUAL_QUESTIONS_PER_SUBTOPIC,
    contest_easy_per_subtopic: int = CONTEST_EASY_QUESTIONS_PER_SUBTOPIC,
    contest_medium_per_subtopic: int = CONTEST_MEDIUM_QUESTIONS_PER_SUBTOPIC,
    contest_hard_per_subtopic: int = CONTEST_HARD_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level_easy: str = CONTEST_LEVEL_EASY,
    contest_level_medium: str = CONTEST_LEVEL_MEDIUM,
    contest_level_hard: str = CONTEST_LEVEL_HARD,
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
            conceptual_per_subtopic,
            contest_easy_per_subtopic,
            contest_medium_per_subtopic,
            contest_hard_per_subtopic,
            tech_active_per_subtopic,
            contest_level_easy,
            contest_level_medium,
            contest_level_hard,
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
    conceptual_per_subtopic: int = CONCEPTUAL_QUESTIONS_PER_SUBTOPIC,
    contest_easy_per_subtopic: int = CONTEST_EASY_QUESTIONS_PER_SUBTOPIC,
    contest_medium_per_subtopic: int = CONTEST_MEDIUM_QUESTIONS_PER_SUBTOPIC,
    contest_hard_per_subtopic: int = CONTEST_HARD_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level_easy: str = CONTEST_LEVEL_EASY,
    contest_level_medium: str = CONTEST_LEVEL_MEDIUM,
    contest_level_hard: str = CONTEST_LEVEL_HARD,
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
            conceptual_per_subtopic,
            contest_easy_per_subtopic,
            contest_medium_per_subtopic,
            contest_hard_per_subtopic,
            tech_active_per_subtopic,
            contest_level_easy,
            contest_level_medium,
            contest_level_hard,
            diagram_mode,
            diagrams_per_subtopic,
        )
        write_output(paths["solutions"], solutions_tex)
    return paths["solutions"]


def generate_worksheets(
    topic: str,
    subtopics: list[str],
    output_dir: Path = OUTPUT_DIR,
    conceptual_per_subtopic: int = CONCEPTUAL_QUESTIONS_PER_SUBTOPIC,
    contest_easy_per_subtopic: int = CONTEST_EASY_QUESTIONS_PER_SUBTOPIC,
    contest_medium_per_subtopic: int = CONTEST_MEDIUM_QUESTIONS_PER_SUBTOPIC,
    contest_hard_per_subtopic: int = CONTEST_HARD_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level_easy: str = CONTEST_LEVEL_EASY,
    contest_level_medium: str = CONTEST_LEVEL_MEDIUM,
    contest_level_hard: str = CONTEST_LEVEL_HARD,
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
        conceptual_per_subtopic,
        contest_easy_per_subtopic,
        contest_medium_per_subtopic,
        contest_hard_per_subtopic,
        tech_active_per_subtopic,
        contest_level_easy,
        contest_level_medium,
        contest_level_hard,
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
            conceptual_per_subtopic,
            contest_easy_per_subtopic,
            contest_medium_per_subtopic,
            contest_hard_per_subtopic,
            tech_active_per_subtopic,
            contest_level_easy,
            contest_level_medium,
            contest_level_hard,
            diagram_mode,
            diagrams_per_subtopic,
        )},
        "questions": lambda: {"worksheet": run_questions_step(
            topic,
            subtopics,
            output_dir,
            conceptual_per_subtopic,
            contest_easy_per_subtopic,
            contest_medium_per_subtopic,
            contest_hard_per_subtopic,
            tech_active_per_subtopic,
            contest_level_easy,
            contest_level_medium,
            contest_level_hard,
            diagram_mode,
            diagrams_per_subtopic,
        )},
        "solutions": lambda: {"solutions": run_solutions_step(
            topic,
            subtopics,
            output_dir,
            conceptual_per_subtopic,
            contest_easy_per_subtopic,
            contest_medium_per_subtopic,
            contest_hard_per_subtopic,
            tech_active_per_subtopic,
            contest_level_easy,
            contest_level_medium,
            contest_level_hard,
            diagram_mode,
            diagrams_per_subtopic,
        )},
        "all": lambda: {
            "research": run_research_step(
                topic,
                subtopics,
                output_dir,
                conceptual_per_subtopic,
                contest_easy_per_subtopic,
                contest_medium_per_subtopic,
                contest_hard_per_subtopic,
                tech_active_per_subtopic,
                contest_level_easy,
                contest_level_medium,
                contest_level_hard,
                diagram_mode,
                diagrams_per_subtopic,
            ),
            "worksheet": run_questions_step(
                topic,
                subtopics,
                output_dir,
                conceptual_per_subtopic,
                contest_easy_per_subtopic,
                contest_medium_per_subtopic,
                contest_hard_per_subtopic,
                tech_active_per_subtopic,
                contest_level_easy,
                contest_level_medium,
                contest_level_hard,
                diagram_mode,
                diagrams_per_subtopic,
            ),
            "solutions": run_solutions_step(
                topic,
                subtopics,
                output_dir,
                conceptual_per_subtopic,
                contest_easy_per_subtopic,
                contest_medium_per_subtopic,
                contest_hard_per_subtopic,
                tech_active_per_subtopic,
                contest_level_easy,
                contest_level_medium,
                contest_level_hard,
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
