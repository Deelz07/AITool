"""Delete auxiliary files from AI_output, keeping only .tex, .txt, and .pdf files."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
AI_OUTPUT_DIR = PROJECT_ROOT / "AI_output"
KEEP_SUFFIXES = {".tex", ".txt", ".pdf"}


def cleanup_ai_output(output_dir: Path = AI_OUTPUT_DIR, dry_run: bool = False) -> list[Path]:
    if not output_dir.exists():
        raise FileNotFoundError(f"Output folder not found: {output_dir}")

    removed: list[Path] = []
    for path in sorted(output_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() in KEEP_SUFFIXES:
            continue

        removed.append(path)
        if dry_run:
            print(f"Would delete: {path.name}")
        else:
            path.unlink()
            print(f"Deleted: {path.name}")

    action = "Would remove" if dry_run else "Removed"
    print(f"\n{action} {len(removed)} file(s) from {output_dir}")
    return removed


if __name__ == "__main__":
    cleanup_ai_output()
