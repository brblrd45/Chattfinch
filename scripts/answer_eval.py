"""Ask the bot every eval question and write a run folder to grade by hand.

    .venv/bin/python scripts/answer_eval.py                 # all 46 questions
    .venv/bin/python scripts/answer_eval.py --name baseline
    .venv/bin/python scripts/answer_eval.py --ids q03 q13   # a few, e.g. after a fix

Writes data/eval_runs/<name>/ with answers.jsonl, sheet.md (read this while grading)
and grades.csv (fill in the blank scores; see eval/RUBRIC.md), then prints the
mechanical checks. Summarize the grades with scripts/score_eval.py.

Costs: one model call per descriptive question (40 of 46; a few cents with Haiku).
Status questions read the private IUCN snapshot, and live ones call eBird, so the
run needs the index, ANTHROPIC_API_KEY and EBIRD_API_KEY.
"""

import argparse
import sys
from datetime import datetime

from chattfinch import evaluation, grading, rag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--name", default=datetime.now().strftime("%Y-%m-%d-%H%M"),
                        help="run folder name under data/eval_runs/ (default: date and time)")
    parser.add_argument("--ids", nargs="+", help="only these question ids")
    args = parser.parse_args()

    questions = evaluation.load_questions()
    if args.ids:
        unknown = set(args.ids) - {q.id for q in questions}
        if unknown:
            print(f"Unknown question ids: {sorted(unknown)}", file=sys.stderr)
            return 1
        questions = [q for q in questions if q.id in args.ids]
    run_dir = grading.RUNS_DIR / args.name
    if run_dir.exists():
        print(f"{run_dir} already exists; pick another --name", file=sys.stderr)
        return 1

    records = []
    for number, question in enumerate(questions, 1):
        try:
            answer, error = rag.answer(question.question), None
        except Exception as failure:  # record it and carry on: one failure shouldn't end the run
            answer, error = None, f"{type(failure).__name__}: {failure}"
        record = grading.make_record(question, answer, error)
        records.append(record)
        mark = "ok " if record.behavior_ok else "BAD"
        print(f"[{number:2}/{len(questions)}] {mark} {question.id} {record.observed}")

    grading.write_run(records, run_dir)
    summary = grading.summarize_grades(grading.load_grades(run_dir / "grades.csv"))
    print(f"\nWrote {run_dir}")
    for name, (passed, applicable) in summary.checks.items():
        print(f"  {name:16} {passed}/{applicable}")
    print(f"\n{summary.ungraded} answers to grade: open sheet.md, score grades.csv "
          "(eval/RUBRIC.md), then run scripts/score_eval.py.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
