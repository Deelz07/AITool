"""Compile a LaTeX file to PDF using latexmk, with a pdflatex fallback."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
AI_OUTPUT_DIR = PROJECT_ROOT / "AI_output"

# --- Edit these before running ---
TEX_PATH = AI_OUTPUT_DIR / "black_scholes_option_pricing_worksheet.tex"
RUN_CLEANUP_AFTER = True


def resolve_tex_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = PROJECT_ROOT / resolved
    if resolved.suffix.lower() != ".tex":
        raise ValueError(f"Expected a .tex file, got: {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(f"TeX file not found: {resolved}")
    return resolved


def compile_tex(tex_path: Path) -> Path:
    pdf_path = tex_path.with_suffix(".pdf")
    work_dir = tex_path.parent

    if shutil.which("latexmk"):
        command = [
            "latexmk",
            "-pdf",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            tex_path.name,
        ]
    elif shutil.which("pdflatex"):
        command = [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            tex_path.name,
        ]
    else:
        raise RuntimeError(
            "Neither latexmk nor pdflatex found. Install TeX Live, e.g. brew install texlive."
        )

    print(f"Compiling: {tex_path}")
    print(f"Command:   {' '.join(command)}")
    print(f"Working directory: {work_dir}\n")

    result = subprocess.run(command, cwd=work_dir)
    if result.returncode != 0:
        log_path = work_dir / tex_path.with_suffix(".log").name
        raise RuntimeError(
            f"LaTeX build failed (exit code {result.returncode}). See log: {log_path}"
        )

    if not pdf_path.exists():
        raise RuntimeError(f"Build finished but PDF not found: {pdf_path}")

    print(f"\nPDF written to: {pdf_path}")
    return pdf_path


def main() -> None:
    tex_path = resolve_tex_path(TEX_PATH)
    compile_tex(tex_path)

    if RUN_CLEANUP_AFTER:
        from cleanup_ai_output import cleanup_ai_output

        print()
        cleanup_ai_output(tex_path.parent)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
