"""Summarize a graded run (see eval/RUBRIC.md).

    .venv/bin/python scripts/score_eval.py data/eval_runs/<name>
    .venv/bin/python scripts/score_eval.py                          # the newest run

Reads <run>/grades.csv and prints the mechanical pass rates and the human scores,
for all questions and by split and name type. Ungraded answers are counted, not scored.
"""

import argparse
import sys
from pathlib import Path

from chattfinch import grading

GROUPS = [
    ("all questions", lambda row: True),
    ("tune", lambda row: row["split"] == "tune"),
    ("holdout", lambda row: row["split"] == "holdout"),
    ("standard names", lambda row: "alias" not in row["tags"].split()),
    ("alternate names", lambda row: "alias" in row["tags"].split()),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run", nargs="?", type=Path,
                        help="run folder (default: the newest under data/eval_runs/)")
    args = parser.parse_args()
    run = args.run
    if run is None:
        runs = sorted(p for p in grading.RUNS_DIR.glob("*") if (p / "grades.csv").exists())
        if not runs:
            print("No runs yet: run scripts/answer_eval.py first", file=sys.stderr)
            return 1
        run = max(runs, key=lambda p: (p / "grades.csv").stat().st_mtime)
    try:
        rows = grading.load_grades(run / "grades.csv")
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Run: {run}\n")
    checks = list(grading.CHECKS)
    print(f"  {'':16} {'n':>3}  " + " ".join(f"{c:>15}" for c in checks)
          + f"  {'graded':>6}  " + " ".join(f"{s:>12}" for s in grading.SCORES))
    for label, keep in GROUPS:
        group = [row for row in rows if keep(row)]
        if not group:
            continue
        s = grading.summarize_grades(group)
        cells = [f"{p}/{a}" if a else "-" for p, a in s.checks.values()]
        scores = [f"{s.means[x]:.2f} ({s.full_marks[x]:.0%})" if s.graded else "-"
                  for x in grading.SCORES]
        print(f"  {label:16} {s.n:>3}  " + " ".join(f"{c:>15}" for c in cells)
              + f"  {s.graded:>6}  " + " ".join(f"{c:>12}" for c in scores))
    print("\nScores: mean on the 0–2 scale (share scoring 2) over graded answers.")
    total = grading.summarize_grades(rows)
    if total.ungraded:
        print(f"⚠ {total.ungraded} answers aren't graded yet; the scores above leave them out.")
    failed = [row for row in rows if "NO" in (row[c] for c in checks)]
    for row in failed:
        print(f"  {row['id']} ({row['split']}): expected {row['expected']}, observed "
              f"{row['observed']}; " + ", ".join(c for c in checks if row[c] == "NO"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
