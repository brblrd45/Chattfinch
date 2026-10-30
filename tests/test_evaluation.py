import json

import pandas as pd
import pytest

from chattfinch import config, corpus, evaluation, router
from chattfinch.evaluation import Question, load_questions, score_retrieval, section_matches
from chattfinch.resolve import Match, Resolution

ROW = {"id": "q01", "type": "behavior", "question": "What does it eat?",
       "expected_species": ["sakfal1"], "expected_sections": ["Ecology"],
       "expect_declined": False, "tags": []}
OUT = {"id": "q02", "type": "out_of_scope", "question": "Penguins?", "expected_species": [],
       "expected_sections": [], "expect_declined": True, "tags": [], "note": "not a KZ bird"}


def write(tmp_path, *rows):
    path = tmp_path / "questions.jsonl"
    path.write_text("\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows) + "\n")
    return path


def test_load_questions(tmp_path):
    first, second = load_questions(write(tmp_path, ROW, "", OUT))  # blank lines are skipped
    assert first == Question(id="q01", type="behavior", question="What does it eat?",
                             expected_species=("sakfal1",), expected_sections=("Ecology",),
                             expect_declined=False)
    assert first.in_scope and not second.in_scope
    assert second.note == "not a KZ bird"
    assert first.split == second.split == "tune"  # the default


def test_load_questions_reads_the_route(tmp_path):
    status = {**ROW, "id": "s", "route": "status", "expected_sections": []}
    asks = {**ROW, "id": "a", "route": "live", "expected_species": [], "expected_sections": [],
            "expect_declined": True}
    first, routed, asking = load_questions(write(tmp_path, ROW, status, asks))
    assert first.route == "descriptive" and first.scored_for_retrieval
    assert routed.route == "status" and routed.in_scope and not routed.scored_for_retrieval
    assert asking.expect_declined and not asking.scored_for_retrieval


def test_routing_results_judge_the_route_and_the_species():
    status = Question("s", "status", "?", ("sakfal1",), (), False, route="status")
    saker = Resolution(matches=(Match("Saker Falcon", "saker falcon", ("sakfal1",), False),))
    (good,) = evaluation.evaluate_routing([status], lambda q: "status", lambda q: saker)
    assert good.route_ok and good.species_ok
    (wrong,) = evaluation.evaluate_routing([status], lambda q: "descriptive",
                                           lambda q: Resolution(matches=()))
    assert not wrong.route_ok and not wrong.species_ok
    descriptive = Question("d", "behavior", "?", ("sakfal1",), (), False)
    (result,) = evaluation.evaluate_routing([descriptive], lambda q: "descriptive",
                                            lambda q: Resolution(matches=()))
    assert result.route_ok and result.species_ok  # judged by retrieval, not here


def test_load_questions_reads_the_split(tmp_path):
    (question,) = load_questions(write(tmp_path, {**ROW, "split": "holdout"}))
    assert question.split == "holdout"


@pytest.mark.parametrize("rows, message", [
    (["{not json"], "line 1: not valid JSON"),
    ([{k: v for k, v in ROW.items() if k != "tags"}], "missing \\['tags'\\]"),
    ([{**ROW, "expected_sectoin": []}], "unknown fields \\['expected_sectoin'\\]"),
    ([ROW, ROW], "line 2: duplicate id 'q01'"),
    ([{**ROW, "type": "diet"}], "type 'diet' is not one of"),
    ([{**ROW, "split": "test"}], "split 'test' is not one of"),
    ([{**ROW, "route": "web"}], "route 'web' is not one of"),
    ([{**ROW, "route": "status"}], "a routed question expects no sections"),
    ([{**ROW, "route": "status", "expected_sections": [], "expect_declined": True}],
     "declined exactly when it names no expected species"),
    ([{**ROW, "route": "live", "expected_species": [], "expected_sections": []}],
     "declined exactly when it names no expected species"),
    ([{**OUT, "route": "status"}], "an out_of_scope question is descriptive"),
    ([{**ROW, "question": "  "}], "empty question"),
    ([{**ROW, "expect_declined": True}], "expect_declined must be true exactly for out_of_scope"),
    ([{**OUT, "expect_declined": False}], "expect_declined must be true exactly for out_of_scope"),
    ([{**OUT, "expected_species": ["sakfal1"]}], "expects no species or sections"),
    ([{**ROW, "expected_species": []}], "needs expected_species"),
])
def test_load_questions_rejects_malformed_rows(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        load_questions(write(tmp_path, *rows))


def test_section_matches_exact_names_and_subsections_only():
    assert section_matches("Feeding ecology", ("Feeding ecology",))
    assert section_matches("Feeding ecology > Diet composition", ("Feeding ecology",))
    assert section_matches("Introduction", ("Status", "Introduction"))
    assert not section_matches("Feeding", ("Feeding ecology",))
    assert not section_matches("Feeding ecology and more", ("Feeding ecology",))
    assert not section_matches("Behaviour > Feeding ecology", ("Feeding ecology",))
    assert not section_matches("Description", ())


def hit(code, section, score=0.5, name="Some Bird"):
    return {"score": score, "metadata": {"species_code": code, "section": section,
                                         "common_name": name}}


QUESTION = Question(id="q", type="behavior", question="?", expected_species=("sakfal1",),
                    expected_sections=("Ecology",), expect_declined=False)


def test_score_retrieval_ranks_species_and_section_separately():
    hits = [hit("reffal1", "Dietary habits", 0.7, "Red-footed Falcon"),
            hit("sakfal1", "Description"), hit("sakfal1", "Ecology")]
    result = score_retrieval(QUESTION, hits)
    assert (result.species_rank, result.section_rank) == (2, 3)
    assert result.reciprocal_rank == 0.5
    assert (result.top_species, result.top_score) == ("Red-footed Falcon", 0.7)


def test_right_section_of_the_wrong_species_does_not_count():
    result = score_retrieval(QUESTION, [hit("reffal1", "Ecology"), hit("sakfal1", "Status")])
    assert (result.species_rank, result.section_rank) == (2, None)


def test_score_retrieval_misses():
    result = score_retrieval(QUESTION, [hit("reffal1", "Ecology")])
    assert result.species_rank is None and result.reciprocal_rank == 0.0
    empty = score_retrieval(QUESTION, [])
    assert empty.species_rank is None and empty.top_species == "" and empty.top_score == 0.0


def test_no_expected_sections_means_no_section_rank():
    question = Question(id="q", type="behavior", question="?", expected_species=("sakfal1",),
                        expected_sections=(), expect_declined=False)
    assert score_retrieval(question, [hit("sakfal1", "Ecology")]).section_rank is None


def test_evaluate_retrieval_summarises_in_scope_questions_only():
    questions = [
        Question("a", "behavior", "first", ("sakfal1",), ("Ecology",), False),
        Question("b", "behavior", "second", ("sakfal1",), ("Ecology",), False),
        Question("c", "behavior", "third", ("sakfal1",), (), False),  # no section label
        Question("d", "behavior", "fourth", ("sakfal1",), ("Ecology",), False),
        Question("e", "out_of_scope", "penguins", (), (), True),
    ]
    answers = {
        "first": [hit("sakfal1", "Ecology")],  # species 1, section 1
        "second": [hit("x", "Ecology"), hit("sakfal1", "Status")],  # species 2, no section
        "third": [hit("x", "y"), hit("x", "y"), hit("x", "y"), hit("sakfal1", "z")],  # species 4
        "fourth": [hit("x", "y")],  # miss
    }
    asked = []

    def retrieve(query, k):
        asked.append((query, k))
        return answers[query]

    summary = evaluation.evaluate_retrieval(questions, retrieve, k=4)
    assert asked == [("first", 4), ("second", 4), ("third", 4), ("fourth", 4)]  # not "penguins"
    assert (summary.n, summary.k) == (4, 4)
    assert summary.hit_at_1 == 0.25 and summary.hit_at_k == 0.75
    assert summary.mrr == pytest.approx((1 + 1 / 2 + 1 / 4 + 0) / 4)
    assert summary.section_hit_at_k == pytest.approx(1 / 3)  # of the 3 with a section label


def test_summarize_scores_a_subset_of_results():
    first = score_retrieval(QUESTION, [hit("sakfal1", "Ecology")])
    second = score_retrieval(QUESTION, [hit("x", "y"), hit("sakfal1", "Ecology")])
    summary = evaluation.summarize([first, second], k=6)
    assert (summary.n, summary.k, summary.hit_at_1, summary.hit_at_k) == (2, 6, 0.5, 1.0)
    assert summary.mrr == pytest.approx(0.75)
    assert summary.results == [first, second]


def test_evaluate_retrieval_with_no_in_scope_questions():
    summary = evaluation.evaluate_retrieval([], lambda query, k: [], k=6)
    assert (summary.n, summary.hit_at_1, summary.mrr, summary.section_hit_at_k) == (0, 0, 0, 0)


# --- The committed eval set -----------------------------------------------------------

@pytest.fixture(scope="module")
def questions():
    return load_questions()


def test_committed_eval_set_is_of_the_planned_size(questions):
    assert 40 <= len(questions) <= 60
    assert len({q.question for q in questions}) == len(questions)  # no repeated questions


def test_committed_eval_set_covers_every_question_type_and_aliases(questions):
    assert {q.type for q in questions} == set(evaluation.TYPES)
    assert sum(not q.in_scope for q in questions) >= 3


def test_committed_eval_set_has_enough_alternate_name_questions_to_measure(questions):
    """Day 4's name resolution is judged on these, in both splits."""
    aliases = [q for q in questions if "alias" in q.tags]
    assert len(aliases) >= 15
    assert sum(q.split == "tune" for q in aliases) >= 8
    assert sum(q.split == "holdout" for q in aliases) >= 5


def test_committed_eval_set_covers_every_route(questions):
    routes = {q.route for q in questions}
    assert routes == set(evaluation.ROUTES)
    assert any(q.route != "descriptive" and q.expect_declined for q in questions)  # asks for a bird
    assert any(q.route != "descriptive" and "alias" in q.tags for q in questions)


def test_committed_eval_set_holds_out_about_a_third(questions):
    holdout = [q for q in questions if q.split == "holdout"]
    assert 0.25 <= len(holdout) / len(questions) <= 0.4
    # Every kind of question is represented, so the held-out totals mean something.
    assert any("alias" in q.tags for q in holdout)
    assert any(q.in_scope and "alias" not in q.tags for q in holdout)
    assert any(not q.in_scope for q in holdout)


def test_expected_species_and_sections_exist_in_the_corpus(questions):
    """A typo in a species code or section name would make a question unanswerable
    and silently drag the metrics down."""
    species = set(pd.read_csv(config.DUMPS_DIR / "species_list.csv")["species_code"])
    sections = {doc["species_code"]: [s["section"] for s in doc["sections"]]
                for doc in corpus.read_corpus(config.DUMPS_DIR / "corpus.jsonl.gz")}
    for q in questions:
        for code in q.expected_species:
            assert code in species, f"{q.id}: unknown species {code}"
        for name in q.expected_sections:
            assert any(section_matches(section, (name,))
                       for code in q.expected_species for section in sections[code]), (
                f"{q.id}: no section {name!r} for {q.expected_species}")


def test_out_of_scope_birds_are_really_absent_from_the_species_list(questions):
    names = pd.read_csv(config.DUMPS_DIR / "species_list.csv")["common_name"].str.lower()
    for bird in ("penguin", "black-necked crane", "himalayan monal", "bald eagle", "red-crowned crane",
                "american kestrel"):
        assert any(bird in q.question.lower() for q in questions if not q.in_scope), bird
        assert not names.str.contains(bird).any(), f"{bird} is on the species list"


def test_load_questions_reads_the_reference_answer(tmp_path):
    (question,) = load_questions(write(tmp_path, {**ROW, "reference_answer": "Ground squirrels."}))
    assert question.reference_answer == "Ground squirrels."
    (without,) = load_questions(write(tmp_path, ROW))
    assert without.reference_answer == ""


def test_every_committed_question_has_a_reference_answer(questions):
    missing = [q.id for q in questions if not q.reference_answer.strip()]
    assert missing == []


def test_reference_answers_name_the_canonical_bird_for_alternate_names(questions):
    """An alias question's reference must say which bird it is, since the answer
    should."""
    names = dict(pd.read_csv(config.DUMPS_DIR / "species_list.csv")[
        ["species_code", "common_name"]].values)
    for q in questions:
        if "alias" in q.tags:
            for code in q.expected_species:
                assert names[code] in q.reference_answer, (q.id, names[code])


def test_status_references_give_no_red_list_category(questions):
    """The eval file is published with the code; IUCN categories must stay out of it."""
    for q in questions:
        if q.route == "status":
            text = q.reference_answer
            for code, full in router.CATEGORIES.items():
                assert f"({code})" not in text and full.lower() not in text.lower(), (q.id, full)
