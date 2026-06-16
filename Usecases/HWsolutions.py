import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import read_files as rf

MODEL = "gemini-2.5-flash"
OUTPUT_DIR = PROJECT_ROOT / "AI_output"

# --- Edit before running ---
INPUT_PATH = PROJECT_ROOT / "input/surds_handout.txt"
OUTPUT_FILE = OUTPUT_DIR / "surds_handout_solutions.txt"


def setup_client() -> genai.Client:
    api_key = os.getenv("GENAI_API_KEY")
    if not api_key:
        raise ValueError(f"GENAI_API_KEY not found. Add it to {PROJECT_ROOT / '.env'}")
    client = genai.Client(api_key=api_key)
    print("Success the client was setup")
    return client


def generate_prompt(preamble: str, instruction: str, data: str) -> str:
    return "\n".join([preamble, instruction, data])


def generate_solutions_worksheet(client: genai.Client, input_path: Path, output_path: Path) -> None:
    preamble = "You are a LaTeX expert. Generate a complete, ready-to-compile .tex file."
    instruction = r"""
    You will be required to read through the input data and convert it into latex format.

    1. If there are any solutions please include them. If not, generate the answer and minimal worked solutions.
    2. Remove any existing questions number such as 3.xx and instead number it from scratch
    3. Remove any excessive formatting from other maths language. E.g. [\color[rgb]{0.35,0.35,0.35}6 = 2+r\] should be converted
        to latex version of that.

    4. Format it such that questions are marked with existing questions numbers and subpart are marked with letters.

    5. Latex Formatting Rules:
        1. Use the 'amsmath' and 'amsfonts' packages.
        2. Ensure all math is wrapped in $...$ for inline and $$...$$ for display.
        3. Output the RAW content only—no markdown wrappers.
    """

    question_file = rf.txtfile_to_string(str(input_path))
    prompt = generate_prompt(preamble=preamble, instruction=instruction, data=question_file)

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="text/plain",
            temperature=0.1,
        ),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.text, encoding="utf-8")
    print(f"Success the latex file was written to {output_path}")


if __name__ == "__main__":
    client = setup_client()
    generate_solutions_worksheet(client, INPUT_PATH, OUTPUT_FILE)
