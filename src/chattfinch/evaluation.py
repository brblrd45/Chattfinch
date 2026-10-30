"""The evaluation set (eval/questions.jsonl), and retrieval and routing metrics over it.

Each line of the file is one question:

    id                 "q01", unique
    type               identification | behavior | range | status | sightings | out_of_scope
    question           the text a user would type
    expected_species   species codes the answer should be about: whose chunks should
                       be retrieved, or whose status / sightings should be looked up
                       ([] if out of scope, or for a routed question naming no bird)
    expected_sections  sections where the answer lives; a name also matches its
                       subsections ("Feeding ecology" matches "Feeding ecology > Diet")
    expect_declined    true if the right behavior is to decline (for a routed question:
                       to ask the user to name a bird)
    tags               e.g. ["alias"] when the question names a species by an alternate
                       name, ["everyday_word"] when it contains a word that is also a
                       bird name ("swift") and must not be read as one
    split              optional: "tune" (the default) or "holdout". Held-out questions
                       measure whether a change generalizes: don't look at their
                       per-question results while tuning retrieval
    route              optional: the path that should answer it (router.py):
                       "descriptive" (the default, retrieval + model), "status" (IUCN
                       snapshot) or "live" (eBird recent sightings). Routed questions
                       have no expected_sections, and are left out of the retrieval
                       metrics: nothing is retrieved for them
    reference_answer   what a correct answer says, for grading answers (day 5). For a
                       descriptive question, a paraphrase of the facts in the expected
                       sections (never copied text); for a declined or routed question,
                       the expected behavior. Status references give no category: that
                       is IUCN data, which can't be redistributed
    note               optional: why the question is here

Ground truth is at the species (and section) level, not the chunk level: chunk
labels would be expensive to write and would break on every re-chunk.

Retrieval metrics here need only the index; routing metrics need nothing at all.
Answer-quality metrics (day 5) build on the same file.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

from chattfinch import config, router

QUESTIONS_PATH = config.ROOT / "eval" / "questions.jsonl"
TYPES = ("identification", "behavior", "range", "status", "sightings", "out_of_scope")
SPLITS = ("tune", "holdout")
ROUTES = (router.DESCRIPTIVE, router.STATUS, router.LIVE)
OPTIONAL = ("split", "route", "reference_answer", "note")
REQUIRED = ("id", "type", "question", "expected_species", "expected_sections",
            "expect_declined", "tags")


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    question: str
    expected_species: tuple[str, ...]
    expected_sections: tuple[str, ...]
    expect_declined: bool
    tags: tuple[str, ...] = ()
    split: str = "tune"
    route: str = router.DESCRIPTIVE
    reference_answer: str = ""
    note: str = ""

    @property
    def in_scope(self) -> bool:
        return not self.expect_declined

    @property
    def scored_for_retrieval(self) -> bool:
        """In scope and answered by retrieval (routed questions retrieve nothing)."""
        return self.in_scope and self.route == router.DESCRIPTIVE


def load_questions(path: Path = QUESTIONS_PATH) -> list[Question]:
    """The eval set, checked for well-formedness. Raises ValueError naming the line
    and the problem."""
    questions, seen = [], set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path.name} line {number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{where}: not valid JSON ({error})") from error
        missing = [key for key in REQUIRED if key not in row]
        if missing:
            raise ValueError(f"{where}: missing {missing}")
        unknown = set(row) - set(REQUIRED) - set(OPTIONAL)
        if unknown:
            raise ValueError(f"{where}: unknown fields {sorted(unknown)}")
        if row["id"] in seen:
            raise ValueError(f"{where}: duplicate id {row['id']!r}")
        seen.add(row["id"])
        if row["type"] not in TYPES:
            raise ValueError(f"{where}: type {row['type']!r} is not one of {TYPES}")
        if row.get("split", "tune") not in SPLITS:
            raise ValueError(f"{where}: split {row['split']!r} is not one of {SPLITS}")
        if not row["question"].strip():
            raise ValueError(f"{where}: empty question")
        route = row.get("route", router.DESCRIPTIVE)
        if route not in ROUTES:
            raise ValueError(f"{where}: route {route!r} is not one of {ROUTES}")
        out_of_scope = row["type"] == "out_of_scope"
        if route != router.DESCRIPTIVE:
            # Routed: no sections (nothing is retrieved). Naming no bird means the right
            # behavior is to ask for one, which counts as a decline.
            if out_of_scope:
                raise ValueError(f"{where}: an out_of_scope question is descriptive")
            if row["expected_sections"]:
                raise ValueError(f"{where}: a routed question expects no sections")
            if row["expect_declined"] != (not row["expected_species"]):
                raise ValueError(f"{where}: a routed question is declined exactly when it "
                                 "names no expected species")
        else:
            if out_of_scope != row["expect_declined"]:
                raise ValueError(f"{where}: expect_declined must be true exactly for out_of_scope")
            if out_of_scope and (row["expected_species"] or row["expected_sections"]):
                raise ValueError(f"{where}: an out_of_scope question expects no species or sections")
            if not out_of_scope and not row["expected_species"]:
                raise ValueError(f"{where}: an in-scope question needs expected_species")
        questions.append(Question(
            id=row["id"], type=row["type"], question=row["question"],
            expected_species=tuple(row["expected_species"]),
            expected_sections=tuple(row["expected_sections"]),
            expect_declined=row["expect_declined"], tags=tuple(row["tags"]),
            split=row.get("split", "tune"), route=route,
            reference_answer=row.get("reference_answer", ""), note=row.get("note", "")))
    return questions


def section_matches(section: str, expected: tuple[str, ...]) -> bool:
    """True if `section` is one of `expected` or a subsection of one."""
    return any(section == name or section.startswith(name + " > ") for name in expected)


@dataclass(frozen=True)
class RetrievalResult:
    """How retrieval did on one in-scope question."""
    question: Question
    species_rank: int | None  # 1-based rank of the first chunk from an expected species
    section_rank: int | None  # ... that is also in an expected section (None if no sections given)
    top_species: str  # common name of the top hit, to see what won instead
    top_score: float

    @property
    def reciprocal_rank(self) -> float:
        return 1.0 / self.species_rank if self.species_rank else 0.0


def score_retrieval(question: Question, hits: list[dict]) -> RetrievalResult:
    """Rank of the first hit from an expected species, and of the first that is
    also from an expected section."""
    species_rank = section_rank = None
    for rank, hit in enumerate(hits, 1):
        meta = hit["metadata"]
        if meta["species_code"] not in question.expected_species:
            continue
        species_rank = species_rank or rank
        if question.expected_sections and section_matches(meta["section"],
                                                          question.expected_sections):
            section_rank = rank
            break
    return RetrievalResult(question=question, species_rank=species_rank,
                           section_rank=section_rank,
                           top_species=hits[0]["metadata"]["common_name"] if hits else "",
                           top_score=hits[0]["score"] if hits else 0.0)


@dataclass(frozen=True)
class RetrievalSummary:
    n: int
    k: int
    hit_at_1: float  # share of questions whose top chunk is from an expected species
    hit_at_k: float  # ... with any of the k chunks from an expected species
    mrr: float  # mean of 1/rank of the first expected-species chunk (0 if absent)
    section_hit_at_k: float  # ... with a chunk from an expected species *and* section
    results: list[RetrievalResult] = field(default_factory=list)


def evaluate_retrieval(questions: list[Question], retrieve, k: int) -> RetrievalSummary:
    """Species-level retrieval metrics over the in-scope questions. `retrieve` has
    retrieval.retrieve's signature."""
    return summarize([score_retrieval(q, retrieve(q.question, k=k))
                      for q in questions if q.scored_for_retrieval], k)


def summarize(results: list[RetrievalResult], k: int) -> RetrievalSummary:
    """Metrics over already-scored results, e.g. one subset of the eval set."""
    n = len(results)
    with_sections = [r for r in results if r.question.expected_sections]
    share = lambda count, total: count / total if total else 0.0
    return RetrievalSummary(
        n=n, k=k,
        hit_at_1=share(sum(r.species_rank == 1 for r in results), n),
        hit_at_k=share(sum(r.species_rank is not None for r in results), n),
        mrr=share(sum(r.reciprocal_rank for r in results), n),
        section_hit_at_k=share(sum(r.section_rank is not None for r in with_sections),
                               len(with_sections)),
        results=results)


@dataclass(frozen=True)
class RoutingResult:
    """Whether one question took its expected path and named its expected birds."""
    question: Question
    route: str  # the path the router chose
    species: tuple[str, ...]  # the species name resolution found

    @property
    def route_ok(self) -> bool:
        return self.route == self.question.route

    @property
    def species_ok(self) -> bool:
        """For routed questions, the lookup must be for exactly the expected birds
        (descriptive questions are judged by retrieval instead)."""
        return (self.question.route == router.DESCRIPTIVE
                or set(self.species) == set(self.question.expected_species))


def evaluate_routing(questions: list[Question], classify, resolver) -> list[RoutingResult]:
    """The router's choice and the resolved species for every question. Free: no
    index, no model, no network. `classify` and `resolver` have router.classify's
    and resolve.resolve's signatures."""
    return [RoutingResult(question=q, route=classify(q.question),
                          species=resolver(q.question).species) for q in questions]
