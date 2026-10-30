"""Answer grading (day 5): run the eval set through the bot, write a sheet to grade
by hand, and summarize the grades. The rubric is eval/RUBRIC.md.

A run is a folder (data/eval_runs/<name>/):

  answers.jsonl  one record per question: the answer, its sources (with their text)
                 and the mechanical checks
  sheet.md       the same, laid out for reading: question, reference, answer, passages
  grades.csv     one row per question: the mechanical checks filled in, the human
                 scores (faithful, complete, cited) blank where a person must grade,
                 or "n/a" where there is nothing to grade

The folder lives under data/ because the sheet quotes Wikipedia passages, and data/
isn't published.
"""

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from chattfinch import citations, config, rag, router
from chattfinch.evaluation import Question

RUNS_DIR = config.DATA_DIR / "eval_runs"
SCORES = ("faithful", "complete", "cited")
SCORE_VALUES = ("0", "1", "2", "n/a")
CHECKS = ("behavior_ok", "route_ok", "citations_valid", "right_bird")
COLUMNS = ("id", "split", "route", "tags", "expected", "observed", *CHECKS, *SCORES, "notes")

ANSWER, DECLINE, ASK = "answer", "decline", "ask"
# An out-of-scope question's reference starts with its expected decline reason.
_DECLINE_REFERENCE = re.compile(r"^Decline \((\w+)\)")


def expected_behavior(question: Question) -> str:
    """"answer"; "decline:<reason>" for out of scope; "ask" for a routed question
    naming no bird."""
    if question.route != router.DESCRIPTIVE and question.expect_declined:
        return ASK
    if question.expect_declined:
        match = _DECLINE_REFERENCE.match(question.reference_answer)
        if not match:
            raise ValueError(f"{question.id}: an out-of-scope reference must start with "
                             "'Decline (<reason>)'")
        return f"{DECLINE}:{match.group(1)}"
    return ANSWER


def observed_behavior(answer: rag.Answer | None, error: str | None) -> str:
    if error is not None:
        return "error"
    if answer.needs_choice:
        return "choice"  # asked which of several birds; not expected for any eval question
    if answer.decline_reason == router.SPECIES_NEEDED:
        return ASK
    if answer.declined:
        return f"{DECLINE}:{answer.decline_reason or 'unspecified'}"
    return ANSWER


@dataclass
class Record:
    """One question's run: what the bot did, and every check."""
    id: str
    split: str
    route: str
    tags: list[str]
    question: str
    reference_answer: str
    expected_species: list[str]
    expected: str
    observed: str
    answer_route: str | None
    text: str
    renamed: list[list[str]]
    sources: list[dict]  # rag.Source fields
    cited: list[int]
    invalid_citations: list[int]
    behavior_ok: bool
    route_ok: bool
    citations_valid: bool
    right_bird: bool | None  # None where it doesn't apply
    error: str | None = None

    @property
    def needs_human_scores(self) -> bool:
        """Descriptive questions the bot rightly answered. Declines, wrong behavior
        and routed answers (written from data) have nothing to grade."""
        return self.behavior_ok and self.expected == ANSWER and self.route == router.DESCRIPTIVE


def make_record(question: Question, answer: rag.Answer | None, error: str | None = None) -> Record:
    expected, observed = expected_behavior(question), observed_behavior(answer, error)
    cited = citations.cited_numbers(answer.text) if answer else []
    invalid = citations.invalid_citations(answer) if answer else []
    cited_species = {s.species_code for s in citations.cited_sources(answer)} if answer else set()
    if expected != ANSWER or observed != ANSWER:
        right_bird = None  # nothing was answered, or nothing should have been
    elif question.route == router.STATUS and not cited:
        right_bird = None  # a species without an IUCN assessment: nothing to cite
    else:
        right_bird = bool(cited_species & set(question.expected_species))
    return Record(
        id=question.id, split=question.split, route=question.route, tags=list(question.tags),
        question=question.question, reference_answer=question.reference_answer,
        expected_species=list(question.expected_species), expected=expected, observed=observed,
        answer_route=answer.route if answer else None, text=answer.text if answer else "",
        renamed=[list(pair) for pair in answer.renamed] if answer else [],
        sources=[asdict(s) for s in answer.sources] if answer else [],
        cited=cited, invalid_citations=invalid,
        behavior_ok=observed == expected,
        route_ok=answer is not None and answer.route == question.route,
        citations_valid=not invalid, right_bird=right_bird, error=error)


def write_run(records: list[Record], run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "answers.jsonl").write_text(
        "".join(json.dumps(asdict(r), ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8")
    (run_dir / "sheet.md").write_text(sheet(records), encoding="utf-8")
    write_grades(records, run_dir / "grades.csv")


def _flag(value: bool | None) -> str:
    return "" if value is None else ("yes" if value else "NO")


def write_grades(records: list[Record], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for r in records:
            scores = "" if r.needs_human_scores else "n/a"
            writer.writerow({"id": r.id, "split": r.split, "route": r.route,
                             "tags": " ".join(r.tags), "expected": r.expected,
                             "observed": r.observed, "behavior_ok": _flag(r.behavior_ok),
                             "route_ok": _flag(r.route_ok),
                             "citations_valid": _flag(r.citations_valid),
                             "right_bird": _flag(r.right_bird),
                             **{score: scores for score in SCORES}, "notes": ""})


def sheet(records: list[Record]) -> str:
    """The run laid out for reading while grading."""
    to_grade = sum(r.needs_human_scores for r in records)
    lines = ["# Answers to grade", "",
             f"{len(records)} questions; **{to_grade} to grade** (marked ✍️). "
             "Score them in grades.csv using eval/RUBRIC.md.", ""]
    for r in records:
        mark = "✍️ " if r.needs_human_scores else ""
        checks = ", ".join(f"{name}: {_flag(getattr(r, name))}" for name in CHECKS
                           if getattr(r, name) is not None)
        lines += [f"## {mark}{r.id} · {r.route} · {r.split}" + (f" · {' '.join(r.tags)}" if r.tags else ""),
                  "", f"**Question:** {r.question}", "",
                  f"**Reference:** {r.reference_answer}", "",
                  f"**Expected:** {r.expected} · **Observed:** {r.observed} · {checks}", ""]
        if r.error:
            lines += [f"**Error:** {r.error}", ""]
        else:
            lines += ["**Answer:**", "", *[f"> {line}" if line else ">" for line in r.text.splitlines()], ""]
        by_number = {s["number"]: s for s in r.sources}
        for number in r.cited:
            source = by_number.get(number)
            if source is None:
                lines += [f"**[{number}]** ⚠️ matches no source given", ""]
                continue
            label = citations.source_label(rag.Source(**source))
            lines += [f"**[{number}]** [{label}]({source['url']})", "",
                      *[f"> {line}" if line else ">" for line in source["text"].splitlines()], ""]
        uncited = [s for s in r.sources if s["number"] not in r.cited]
        if uncited:
            lines += ["Retrieved but not cited: " + "; ".join(
                f"[{s['number']}] {s['common_name']} — {s['section']}" for s in uncited), ""]
    return "\n".join(lines)


def load_grades(path: Path) -> list[dict]:
    """The graded CSV, checked: scores are 0, 1, 2, n/a or blank (not graded yet).
    Raises ValueError naming the row and column."""
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for number, row in enumerate(rows, 2):  # line 1 is the header
        for score in SCORES:
            value = (row.get(score) or "").strip()
            if value and value not in SCORE_VALUES:
                raise ValueError(f"{path.name} line {number} ({row['id']}): {score} is "
                                 f"{value!r}; use 0, 1, 2 or n/a")
            row[score] = value
    return rows


@dataclass(frozen=True)
class GradeSummary:
    n: int  # questions in this group
    checks: dict[str, tuple[int, int]]  # check -> (passed, applicable)
    graded: int  # answers with all three scores
    ungraded: int  # answers still waiting for a score
    means: dict[str, float]  # score -> mean over graded answers
    full_marks: dict[str, float]  # score -> share of graded answers scoring 2


def summarize_grades(rows: list[dict]) -> GradeSummary:
    checks = {}
    for name in CHECKS:
        applicable = [row for row in rows if row[name] in ("yes", "NO")]
        checks[name] = (sum(row[name] == "yes" for row in applicable), len(applicable))
    to_grade = [row for row in rows if any(row[s] != "n/a" for s in SCORES)]
    graded = [row for row in to_grade if all(row[s] in ("0", "1", "2") for s in SCORES)]
    share = lambda count: count / len(graded) if graded else 0.0
    return GradeSummary(
        n=len(rows), checks=checks, graded=len(graded), ungraded=len(to_grade) - len(graded),
        means={s: share(sum(int(row[s]) for row in graded)) for s in SCORES},
        full_marks={s: share(sum(row[s] == "2" for row in graded)) for s in SCORES})
