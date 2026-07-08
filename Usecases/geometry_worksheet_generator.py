"""
Generate geometry homework worksheets with TikZ/PGF diagrams for every question.

Same pipeline as HW_worksheet_generator_og.py (research → worksheet → solutions),
but diagram mode is always on and every question must include a compile-ready TikZ
figure — never external images, \\includegraphics, or AI-generated picture files.

Each LLM call can be run separately via STEP. Intermediate files are saved under
AI_output/ so a failed step can be retried without redoing earlier work.
"""

import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import generate_content as AI
from google.genai import types

# Reuse battle-tested infrastructure from the OG generator.
from Usecases import HW_worksheet_generator_og as og

# --- Edit these before running ---
MODEL = og.MODEL
OUTPUT_DIR = og.OUTPUT_DIR
PAST_QUERIES_FILE = og.PAST_QUERIES_FILE
LOG_FILE = OUTPUT_DIR / "geometry_worksheet_generator.log"

TOPIC = "Geometry_test"
SUBTOPICS = [
    "Vertically opposite angles",
]

CORE_QUESTIONS_PER_SUBTOPIC = 4
CHALLENGE_QUESTIONS_PER_SUBTOPIC = 3
TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC = 1
CONTEST_LEVEL = "AMC 8"
STEP = "all"

USE_BATCH_API = og.USE_BATCH_API
BATCH_POLL_INTERVAL_SEC = og.BATCH_POLL_INTERVAL_SEC
MAX_OUTPUT_TOKENS = og.MAX_OUTPUT_TOKENS
SOLUTIONS_CONTINUATION_MAX = og.SOLUTIONS_CONTINUATION_MAX
SOLUTIONS_MIN_CHARS_RATIO = og.SOLUTIONS_MIN_CHARS_RATIO

# Every question in every section must carry its own diagram.
DIAGRAMS_PER_SUBTOPIC = CORE_QUESTIONS_PER_SUBTOPIC

logger = og.logger

GEOMETRY_REPUTABLE_SOURCES = """
Prioritise content from reputable geometry and contest sources such as:
- Art of Problem Solving (AOPS) geometry wikis and forums
- UKMT past papers and mentoring sheets (Junior, Intermediate, Senior)
- AMC 8 / AMC 10 / AIME geometry-style problems (write originals only)
- Khan Academy, OpenStax, and official exam-board geometry specs (AQA, Edexcel, OCR)
- Wolfram MathWorld for definitions; Cut-the-Knot and geometry references for classical results
- Euclidean geometry texts and olympiad geometry notes (for challenge insight, not copying)
Never copy existing contest questions verbatim.
"""

GEOMETRY_CONTEST_STYLE = """
Geometry contest-style challenge problems should feel like original UKMT/AMC geometry:
- Difficulty target: {contest_level}.
- The diagram is part of the problem — the figure must contain exactly the givens, no extra hints.
- Reward angle chasing, similar triangles, cyclic quadrilaterals, power of a point, or area ratios
  when appropriate to the subtopic.
- Prefer exact answers (degrees with justification, surds, simplified ratios, or integer lengths).
- Use precise wording: "Find the measure of...", "Prove that...", "Determine the ratio...".
- Avoid coordinate-bash-only problems unless the subtopic calls for coordinate geometry.
- Each problem should be solvable in contest time once the key relationship is spotted.
- No calculator assumed for challenge section.
"""

NO_EXTERNAL_IMAGES = """
CRITICAL — NO EXTERNAL IMAGES:
- Do NOT use \\includegraphics, \\graphicx, SVG/PNG/JPG links, or placeholders for figures.
- Do NOT write "see diagram below" without providing the full TikZ code inline.
- Every figure must be native LaTeX/TikZ/PGF built from coordinates, paths, and labels.
- Diagrams are mathematical constructions, not illustrations or clip art.
"""

GEOMETRY_TIKZ_LAYOUT = """
TIKZ LAYOUT AND LABELLING (mandatory — diagram quality depends on this)

Coordinate strategy:
1. Pick a key point first (intersection O, vertex P, circle centre O). Place it at (0,0) unless
   the figure needs offset.
2. Define EVERY \\coordinate before any \\draw, \\fill, or \\pic that references it.
3. For lines through a point, use polar rays so drawn angles match the givens numerically:
   \\coordinate (A) at (0:2.5);   % ray along 0° (positive x-axis)
   \\coordinate (B) at (180:2.5); % opposite ray
   \\coordinate (C) at (47:2.5);  % second line at 47° — the arc from ray OA to ray OC is 47°
4. Extend each line as \\draw (A) -- (O) -- (B); so the vertex sits between the two arms shown.
5. For a given angle of θ° between two rays from O, put the second ray at polar angle θ (or 180-θ
   as appropriate). Do NOT use arbitrary slopes that contradict the stated angle.

Angle marks (always use the angles library — never floating \\node labels for degrees):
\\pic [draw, "$47^\\circ$", angle radius=7mm, angle eccentricity=1.25] {angle = A--O--C};
- Syntax: {angle = <arm1>--<vertex>--<arm2>} — vertex in the middle, arms in counterclockwise order.
- Define arm points before the \\pic line.
- Use angle radius=6–10mm and angle eccentricity=1.2–1.5 so labels sit outside the arc.
- Mark ONLY the angles named in the question; do not duplicate the same label twice.

Vertex labels (points, not angle measures):
- Place with directional anchors: \\node[above left] at (A) {$A$};
- Or: \\coordinate (A) at (130:2); \\node[label={above left:$A$}] at (A) {};
- Do NOT wrap labels in double braces like {{$A$}} — use {$A$} only.
- Keep labels away from arcs and edges; label line names ($L_1$, $k$) at ray endpoints.

Line intersections (calc library — preferred for two lines):
\\coordinate (P) at (intersection of A--E and B--D);
- Define all four points A, E, B, D first; then one \\coordinate for P.
- Do NOT use \\path[name intersections={...}] alone — it often fails to define the point.

Circle chords that cross inside the circle:
- Place four points in ALTERNATING order around the circle (e.g. A=10°, C=100°, B=190°, D=280°).
- If all four points lie on one semicircle, chords AB and CD will not cross — this causes a compile error.
\\path[name path=chordAB] (A) -- (B);
\\path[name path=chordCD] (C) -- (D);
\\fill[name intersections={of=chordAB and chordCD, by=P}] (P) circle (0.5pt);
- Verify in the plan comment that going around the circle the order is A, C, B, D (alternating chords).

Anti-patterns (NEVER do these):
- Markdown code fences (```latex or ```) around the document.
- \\begin{enumerate}[label=(\\alpha)] — use [label=(\\alph*)] for subparts (a), (b), (c).
- \\node at (-1.8, 0) {$45^\\circ$} without an angle arc — labels drift and look wrong.
- \\pic {angle = B--P--A} before \\coordinate (A) or (B) is defined.
- \\path[name intersections={of=..., by=P}] without \\fill — P may be undefined.
- Chord endpoints on the same arc (non-alternating) — intersection of chords fails.
- \\url{...} in Sources without \\usepackage{url} in the preamble.
- Double-brace node labels {{$A$}} — use {$A$}.

Canonical template — two lines intersecting at E, angle AEC = 47° (adapt for each question):
% Diagram plan: rays EA (180°), EB (0°), EC (47°), ED (227°) from E; mark angle AEC = 47°
\\begin{center}
\\begin{tikzpicture}[scale=1.0]
    \\coordinate (E) at (0,0);
    \\coordinate (A) at (180:2.4);
    \\coordinate (B) at (0:2.4);
    \\coordinate (C) at (47:2.2);
    \\coordinate (D) at (227:2.2);
    \\draw[thick] (A) -- (B);
    \\draw[thick] (C) -- (D);
    \\node[below left] at (E) {$E$};
    \\node[left] at (A) {$A$};
    \\node[right] at (B) {$B$};
    \\node[above] at (C) {$C$};
    \\node[below] at (D) {$D$};
    \\pic [draw, "$47^\\circ$", angle radius=7mm, angle eccentricity=1.3] {angle = A--E--C};
\\end{tikzpicture}
\\end{center}

For n lines through one point with consecutive angles θ₁, θ₂, …:
- Place first ray at 0°, second at θ₁°, third at (θ₁+θ₂)°, etc. (cumulative from positive x-axis).
- Mark each given angle with a separate \\pic using the correct consecutive arm pairs.

Before writing each diagram, state in the plan comment the polar angles (in degrees) of every ray.
"""

GEOMETRY_DIAGRAM_STYLE = """
GEOMETRY DIAGRAM MODE — TikZ REQUIRED FOR EVERY QUESTION

{no_images}

{layout_guide}

Use TikZ for ALL figures. Before each diagram, write a one-line plan comment:
% Diagram plan: <shape type, ray angles in degrees, givens, labels, what is marked>

TikZ preamble (include in every .tex document):
\\usepackage{tikz}
\\usetikzlibrary{angles, quotes, calc, positioning, arrows.meta, intersections, patterns}

Drawing rules:
1. Follow the layout guide above: polar rays, all coordinates defined first, \\pic for every angle label.
2. Wrap each figure in \\begin{center}\\begin{tikzpicture}[scale=0.9] ... \\end{tikzpicture}\\end{center}
3. If a question states an angle measure, the TikZ coordinates MUST produce that angle visually.
4. Mark given equal lengths with matching tick marks; right angles with the angles library square mark.
5. Draw only what the question states — no solution hints in the question diagram.
6. Keep figures readable: one main configuration, at most one or two auxiliary dashed lines.
7. For circles use \\draw (O) circle (r); place chord endpoints in alternating order around the circle.
8. Avoid 3D, perspective, pgfplots surfaces, or overlapping clutter.

Question coverage (strict):
- EVERY core, challenge, and tech-active question MUST include its own TikZ diagram
  immediately before or after the question text (at least one per \\item).
- Section A: at least {diagrams_per_subtopic} diagram-backed questions per subtopic (all {core_per_subtopic} if possible).
- Section B: each challenge problem includes a diagram showing the contest setup.
- Section C: each tech-active question includes a diagram of the geometric configuration;
  CAS work may follow, but the figure is still TikZ in LaTeX.

Self-check each diagram before output:
- [ ] Every \\coordinate used in \\pic or \\draw is defined above it in the same tikzpicture.
- [ ] Every stated angle has a \\pic arc (not a floating \\node).
- [ ] Ray directions match the numeric givens (polar angles sum correctly at the vertex).
- [ ] Vertex labels sit on the correct points and do not overlap arcs or edges.
- [ ] Compiles without TikZ errors; labels are readable at scale=0.9–1.2.
"""

GEOMETRY_LATEX_RULES = og.LATEX_RULES + """
6. Include \\usepackage{tikz} and \\usetikzlibrary{angles, quotes, calc, positioning, arrows.meta, intersections, patterns}.
7. Include \\usepackage{enumitem} for clean lists; subparts use [label=(\\alph*)], never [label=(\\alpha)].
8. Include \\usepackage{url} whenever the Sources section uses \\url{...}.
9. Never use \\includegraphics or image files for figures.
10. Output RAW LaTeX only — no markdown code fences (no ```latex or ``` wrappers).
11. Node labels use {$A$}, not {{$A$}}; every \\begin{{document}} must have matching \\end{{document}}.
"""

GEOMETRY_SOLUTION_DIAGRAMS = """
10. For EVERY question, include the original TikZ diagram from the worksheet (reproduce verbatim).
11. When auxiliary lines clarify the proof (altitudes, radii, angle bisectors, extra chords),
    add a second "Solution diagram" using dashed grey lines (e.g. draw[dashed, gray] ...).
12. Do not replace TikZ with prose descriptions of the figure.
13. When redrawing diagrams, follow the same layout rules: polar rays, coordinates before \\pic,
    \\pic for angle labels with angle radius and angle eccentricity — no floating degree nodes.
"""


def geometry_diagram_block(
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
    core_per_subtopic: int = CORE_QUESTIONS_PER_SUBTOPIC,
) -> str:
    return GEOMETRY_DIAGRAM_STYLE.format(
        no_images=NO_EXTERNAL_IMAGES,
        layout_guide=GEOMETRY_TIKZ_LAYOUT,
        diagrams_per_subtopic=diagrams_per_subtopic,
        core_per_subtopic=core_per_subtopic,
    )


def sanitize_geometry_tex(content: str) -> str:
    """Fix common LLM LaTeX mistakes before write/compile."""
    cleaned = og.strip_markdown_fences(content)

    cleaned = cleaned.replace(
        r"\begin{enumerate}[label=(\alpha)]",
        r"\begin{enumerate}[label=(\alph*)]",
    )
    cleaned = re.sub(r"\{\{(\$[^}]+\$)\}\}", r"{\1}", cleaned)

    if r"\url{" in cleaned and r"\usepackage{url}" not in cleaned:
        if r"\usepackage{enumitem}" in cleaned:
            cleaned = cleaned.replace(
                r"\usepackage{enumitem}",
                r"\usepackage{enumitem}" + "\n\\usepackage{url}",
                1,
            )
        elif r"\usepackage{tikz}" in cleaned:
            cleaned = cleaned.replace(
                r"\usepackage{tikz}",
                r"\usepackage{url}" + "\n\\usepackage{tikz}",
                1,
            )

    # Bare \path[name intersections=...] without \fill rarely defines the point — upgrade pattern.
    cleaned = re.sub(
        r"\\path\[name intersections=\{of=([^}]+), by=([^}]+)\}\];",
        r"\\fill[name intersections={of=\1, by=\2}] (\2) circle (0.5pt);",
        cleaned,
    )

    if cleaned.rstrip().endswith(r"\end{document}"):
        return cleaned.rstrip() + "\n"
    if r"\begin{document}" in cleaned and r"\end{document}" not in cleaned:
        cleaned = cleaned.rstrip() + "\n\\end{document}\n"
    return cleaned


def write_tex_output(path: Path, content: str) -> None:
    if path.suffix.lower() == ".tex":
        content = sanitize_geometry_tex(content)
    og.write_output(path, content)


def geometry_research_topic(
    client,
    topic: str,
    subtopics: list[str],
    core_per_subtopic: int = CORE_QUESTIONS_PER_SUBTOPIC,
    challenge_per_subtopic: int = CHALLENGE_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level: str = CONTEST_LEVEL,
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
) -> str:
    cleaned = og.clean_subtopics(subtopics)
    subtopic_list = og.format_subtopics(cleaned)
    distribution = og.format_distribution(
        cleaned, core_per_subtopic, challenge_per_subtopic, tech_active_per_subtopic
    )
    contest_style = GEOMETRY_CONTEST_STYLE.format(contest_level=contest_level)
    diagram_block = geometry_diagram_block(diagrams_per_subtopic, core_per_subtopic)

    prompt = f"""You are an experienced geometry curriculum designer and contest problem writer.

Research the geometry topic below using reputable online educational resources.
{GEOMETRY_REPUTABLE_SOURCES}

Topic: {topic}

Subtopics to cover:
{subtopic_list}

{distribution}
{diagram_block}

Produce structured research notes that include:
1. Key definitions, theorems, and techniques for each subtopic.
2. Typical diagram types (triangles, circles, polygons, angle marks) and difficulty progression.
3. Common misconceptions and diagram-reading errors students make.
4. For each subtopic separately, exactly {core_per_subtopic} core practice question ideas.
   Each idea MUST include a plain-text TikZ diagram plan listing:
   - Key point(s) and every ray's polar angle in degrees from the positive x-axis
   - Which angles are marked and their measures
   - Vertex labels and where each sits relative to its point
   - What the student must find
5. For each subtopic separately, exactly {challenge_per_subtopic} contest-style challenge ideas.
{contest_style}
   For each, note: problem sketch, TikZ diagram plan, key geometric insight, and difficulty fit.
6. For each subtopic separately, exactly {tech_active_per_subtopic} Tech Active question ideas.
{og.TECH_ACTIVE_STYLE}
   Each must describe a geometric configuration (with diagram plan) where technology assists
   measurement, tracing, or verification — not replace the need for a TikZ figure in LaTeX.
7. A short list of sources consulted (name and URL where available).
8. TikZ pitfalls to avoid: undefined coordinates before \\pic, floating degree labels without
   arcs, polar angles that contradict stated measures, duplicate angle marks, wrong vertex labels.

Write clear plain-text notes only. Do not output LaTeX yet.
"""

    return og.call_model(
        client,
        "geometry_research",
        prompt,
        model=MODEL,
        contents=prompt,
        config=og.generation_config(
            tools=[types.Tool(google_search=types.GoogleSearch())],
            temperature=0.25,
        ),
    )


def geometry_generate_questions_worksheet(
    client,
    topic: str,
    subtopics: list[str],
    research_notes: str,
    core_per_subtopic: int = CORE_QUESTIONS_PER_SUBTOPIC,
    challenge_per_subtopic: int = CHALLENGE_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level: str = CONTEST_LEVEL,
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
) -> str:
    cleaned = og.clean_subtopics(subtopics)
    num_core = og.total_questions(cleaned, core_per_subtopic)
    num_challenge = og.total_questions(cleaned, challenge_per_subtopic)
    num_tech_active = og.total_questions(cleaned, tech_active_per_subtopic)
    distribution = og.format_distribution(
        cleaned, core_per_subtopic, challenge_per_subtopic, tech_active_per_subtopic
    )
    subtopic_list = og.format_subtopics(cleaned)
    contest_style = GEOMETRY_CONTEST_STYLE.format(contest_level=contest_level)
    diagram_block = geometry_diagram_block(diagrams_per_subtopic, core_per_subtopic)

    preamble = (
        "You are a LaTeX expert, Euclidean geometry specialist, and contest problem writer. "
        "Generate a complete, ready-to-compile .tex file with accurate TikZ diagrams for every "
        "question. Diagram angles and labels must match the givens geometrically — use polar "
        "coordinates and the angles library, never approximate with arbitrary slopes or floating "
        "degree nodes."
    )
    instruction = f"""
Create a geometry homework worksheet for the topic "{topic}" using the research notes below.

Subtopics:
{subtopic_list}

{distribution}
{diagram_block}

Requirements:
1. Write original questions only — do not include solutions or answers.
2. Structure the worksheet into three clearly separated sections:

   **Section A — Core Practice**
   - Use a LaTeX subsection for each subtopic (in the order listed above).
   - Exactly {core_per_subtopic} core questions per subtopic; number 1 to {num_core} consecutively.
   - Each \\item MUST include a TikZ diagram for the geometric configuration.

   **Section B — Challenge Problems (Contest Style)**
   - Section title: "Challenge Problems (Contest Style)".
   - Subsection per subtopic; exactly {challenge_per_subtopic} problems per subtopic.
   - Number C1 to C{num_challenge}; each with a contest-style TikZ diagram.
{contest_style}

   **Section C — Tech Active (CAS Calculator Required)**
   - Section title: "Tech Active (CAS Calculator Required)".
   - State: "A CAS calculator is required for this section."
   - Subsection per subtopic; exactly {tech_active_per_subtopic} questions per subtopic.
   - Number T1 to T{num_tech_active}; each with a TikZ diagram of the setup.
{og.TECH_ACTIVE_STYLE}

3. Add a worksheet title matching the topic.
4. End with a "Sources consulted" bullet list.

{GEOMETRY_LATEX_RULES}
"""
    prompt = AI.generate_prompt(preamble=preamble, instruction=instruction, data=research_notes)

    return sanitize_geometry_tex(
        og.call_model(
            client,
            "geometry_questions_worksheet",
            prompt,
            model=MODEL,
            contents=prompt,
            config=og.generation_config(
                response_mime_type="text/plain",
                temperature=0.08,
            ),
        )
    )


def geometry_generate_solutions_worksheet(
    client,
    topic: str,
    subtopics: list[str],
    questions_worksheet: str,
    core_per_subtopic: int = CORE_QUESTIONS_PER_SUBTOPIC,
    challenge_per_subtopic: int = CHALLENGE_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level: str = CONTEST_LEVEL,
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
) -> str:
    cleaned = og.clean_subtopics(subtopics)
    num_core = og.total_questions(cleaned, core_per_subtopic)
    num_challenge = og.total_questions(cleaned, challenge_per_subtopic)
    num_tech_active = og.total_questions(cleaned, tech_active_per_subtopic)
    diagram_block = geometry_diagram_block(diagrams_per_subtopic, core_per_subtopic)

    preamble = (
        "You are a LaTeX expert and geometry solutions writer. "
        "Generate a complete, ready-to-compile .tex file with TikZ diagrams."
    )
    instruction = f"""
Create worked solutions for the geometry topic "{topic}" using the worksheet below.
Do not invent, reword, or substitute different questions.
{diagram_block}

Requirements:
1. Mirror the worksheet structure exactly (Sections A, B, C and all subtopic subsections).
2. Solve core questions 1–{num_core}, challenge C1–C{num_challenge}, tech-active T1–T{num_tech_active}.
3. For each question: label, \\textbf{{Solution:}}, then working.
4. Challenge problems ({contest_level}): start with "Key idea:" then a clean proof path.
5. Tech Active: start with "CAS approach:"; still include TikZ diagrams from the worksheet.
6. Title: "{topic} — Worked Solutions".
7. Do not include the worksheet's "Sources consulted" section.
{GEOMETRY_SOLUTION_DIAGRAMS}

{GEOMETRY_LATEX_RULES}
"""
    prompt = AI.generate_prompt(
        preamble=preamble,
        instruction=instruction,
        data=questions_worksheet,
    )
    expected_questions = num_core + num_challenge + num_tech_active
    config = og.generation_config(
        response_mime_type="text/plain",
        temperature=0.1,
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
            "geometry_solutions_worksheet"
            if part_index == 0
            else f"geometry_solutions_worksheet_cont_{part_index + 1}"
        )
        part_text, response = og.invoke_model(
            client,
            call_name,
            continuation_prompt,
            **request_kwargs,
        )
        if not part_text.strip():
            raise RuntimeError(f"{call_name} returned empty text.")

        parts.append(part_text)
        merged = og.join_solution_parts(parts)
        truncated = response is not None and og.response_hit_token_limit(response)
        complete = og.solutions_document_complete(merged) and not truncated

        logger.info(
            "Solutions part %d: %d chars (merged total %d chars, complete=%s, truncated=%s)",
            part_index + 1,
            len(part_text),
            len(merged),
            complete,
            truncated,
        )

        if complete:
            og.validate_solutions_document(merged, questions_worksheet, expected_questions)
            return sanitize_geometry_tex(merged)

        if part_index >= SOLUTIONS_CONTINUATION_MAX:
            break

        tail = merged[-2000:]
        continuation_prompt = og.build_solutions_continuation_prompt(
            topic,
            num_core,
            num_challenge,
            num_tech_active,
            tail,
        )
        request_kwargs["contents"] = continuation_prompt

    merged = og.join_solution_parts(parts)
    if og.solutions_document_complete(merged):
        og.validate_solutions_document(merged, questions_worksheet, expected_questions)
        return sanitize_geometry_tex(merged)

    raise og.TruncatedResponseError(
        f"Geometry solutions still incomplete after {len(parts)} part(s) "
        f"({len(merged)} chars). Try STEP='solutions' again."
    )


def log_geometry_run_config(
    topic: str,
    subtopics: list[str],
    step: str,
    output_dir: Path,
    core_per_subtopic: int,
    challenge_per_subtopic: int,
    tech_active_per_subtopic: int,
    contest_level: str,
    diagrams_per_subtopic: int,
) -> None:
    og.log_run_config(
        topic,
        subtopics,
        step,
        output_dir,
        core_per_subtopic,
        challenge_per_subtopic,
        tech_active_per_subtopic,
        contest_level,
        diagram_mode=True,
        diagrams_per_subtopic=diagrams_per_subtopic,
    )
    logger.info("  Generator: geometry_worksheet_generator (TikZ-only, no external images)")


def run_research_step(
    topic: str,
    subtopics: list[str],
    output_dir: Path = OUTPUT_DIR,
    core_per_subtopic: int = CORE_QUESTIONS_PER_SUBTOPIC,
    challenge_per_subtopic: int = CHALLENGE_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level: str = CONTEST_LEVEL,
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
) -> Path:
    cleaned = og.clean_subtopics(subtopics)
    if not cleaned:
        raise ValueError("At least one subtopic is required.")

    og.log_past_query(topic, cleaned)
    paths = og.topic_paths(topic, output_dir)

    with og.log_step("research"):
        client = AI.setup_client()
        logger.info("Gemini client initialised")
        research_notes = geometry_research_topic(
            client,
            topic,
            cleaned,
            core_per_subtopic,
            challenge_per_subtopic,
            tech_active_per_subtopic,
            contest_level,
            diagrams_per_subtopic,
        )
        og.write_output(paths["research"], research_notes)
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
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
) -> Path:
    cleaned = og.clean_subtopics(subtopics)
    if not cleaned:
        raise ValueError("At least one subtopic is required.")

    paths = og.topic_paths(topic, output_dir)
    research_notes = og.read_step_input(paths["research"], "research")

    with og.log_step("questions"):
        client = AI.setup_client()
        logger.info("Gemini client initialised")
        questions_tex = geometry_generate_questions_worksheet(
            client,
            topic,
            cleaned,
            research_notes,
            core_per_subtopic,
            challenge_per_subtopic,
            tech_active_per_subtopic,
            contest_level,
            diagrams_per_subtopic,
        )
        write_tex_output(paths["worksheet"], questions_tex)
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
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
) -> Path:
    cleaned = og.clean_subtopics(subtopics)
    if not cleaned:
        raise ValueError("At least one subtopic is required.")

    paths = og.topic_paths(topic, output_dir)
    questions_tex = og.read_step_input(paths["worksheet"], "worksheet")

    with og.log_step("solutions"):
        client = AI.setup_client()
        logger.info("Gemini client initialised")
        solutions_tex = geometry_generate_solutions_worksheet(
            client,
            topic,
            cleaned,
            questions_tex,
            core_per_subtopic,
            challenge_per_subtopic,
            tech_active_per_subtopic,
            contest_level,
            diagrams_per_subtopic,
        )
        write_tex_output(paths["solutions"], solutions_tex)
    return paths["solutions"]


def generate_worksheets(
    topic: str,
    subtopics: list[str],
    output_dir: Path = OUTPUT_DIR,
    core_per_subtopic: int = CORE_QUESTIONS_PER_SUBTOPIC,
    challenge_per_subtopic: int = CHALLENGE_QUESTIONS_PER_SUBTOPIC,
    tech_active_per_subtopic: int = TECH_ACTIVE_QUESTIONS_PER_SUBTOPIC,
    contest_level: str = CONTEST_LEVEL,
    diagrams_per_subtopic: int = DIAGRAMS_PER_SUBTOPIC,
    step: str = STEP,
) -> dict[str, Path]:
    og.setup_logging(LOG_FILE)
    log_geometry_run_config(
        topic,
        subtopics,
        step,
        output_dir,
        core_per_subtopic,
        challenge_per_subtopic,
        tech_active_per_subtopic,
        contest_level,
        diagrams_per_subtopic,
    )
    logger.warning(
        "Geometry generator — every question requires TikZ; verify PDFs compile cleanly."
    )
    if USE_BATCH_API:
        logger.warning(
            "Batch API ON — 50%% token discount, but each call is async and may take minutes."
        )

    run_start = time.perf_counter()
    steps = {
        "research": lambda: {
            "research": run_research_step(
                topic,
                subtopics,
                output_dir,
                core_per_subtopic,
                challenge_per_subtopic,
                tech_active_per_subtopic,
                contest_level,
                diagrams_per_subtopic,
            )
        },
        "questions": lambda: {
            "worksheet": run_questions_step(
                topic,
                subtopics,
                output_dir,
                core_per_subtopic,
                challenge_per_subtopic,
                tech_active_per_subtopic,
                contest_level,
                diagrams_per_subtopic,
            )
        },
        "solutions": lambda: {
            "solutions": run_solutions_step(
                topic,
                subtopics,
                output_dir,
                core_per_subtopic,
                challenge_per_subtopic,
                tech_active_per_subtopic,
                contest_level,
                diagrams_per_subtopic,
            )
        },
        "all": lambda: {
            "research": run_research_step(
                topic,
                subtopics,
                output_dir,
                core_per_subtopic,
                challenge_per_subtopic,
                tech_active_per_subtopic,
                contest_level,
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
                diagrams_per_subtopic,
            ),
        },
    }

    if step not in steps:
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
        generate_worksheets(
            topic=TOPIC,
            subtopics=SUBTOPICS,
            output_dir=OUTPUT_DIR,
            step=STEP,
        )
    except Exception:
        logger.error("Run aborted due to error.")
        raise
