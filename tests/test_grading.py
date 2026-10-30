"""Answer grading: the mechanical checks, the run files and the grade summary."""

import csv

import pytest

from chattfinch import evaluation, grading, rag, router
from chattfinch.evaluation import Question
from test_telegram_format import source

SAKER = Question("q03", "behavior", "What does the Saker Falcon eat?", ("sakfal1",), ("Ecology",),
                 False, reference_answer="Rodents and birds.")
PENGUINS = Question("q16", "out_of_scope", "Tell me about penguins.", (), (), True,
                    reference_answer="Decline (bird_not_found): not a Kazakhstan bird.")
CAPITAL = Question("q19", "out_of_scope", "Capital of Kazakhstan?", (), (), True,
                   reference_answer="Decline (not_about_birds): not about birds.")
OWLS = Question("q46", "status", "Are owls endangered?", (), (), True, route="status",
                reference_answer="Asks the user to name one bird.")
SWAN_GOOSE = Question("q43", "status", "Is the Swan Goose on the Red List?", ("swagoo1",), (),
                      False, route="status", reference_answer="Says there is no assessment.")


def answered(text="Rodents [1].", sources=None, **fields):
    return rag.Answer(query="q", text=text, declined=False,
                      sources=sources if sources is not None else [source(1)], **fields)


def declined(reason, **fields):
    return rag.Answer(query="q", text="No.", declined=True, sources=[], decline_reason=reason,
                      **fields)


def test_a_good_answer_passes_every_check_and_awaits_human_scores():
    record = grading.make_record(SAKER, answered())
    assert (record.expected, record.observed) == ("answer", "answer")
    assert record.behavior_ok and record.route_ok and record.citations_valid and record.right_bird
    assert record.needs_human_scores


def test_an_invented_citation_and_the_wrong_bird_are_caught():
    other = rag.Source(**{**source(1).__dict__, "species_code": "reffal1"})
    record = grading.make_record(SAKER, answered("Beetles [1]. Mice [4].", [other]))
    assert record.invalid_citations == [4] and not record.citations_valid
    assert record.right_bird is False


def test_declines_must_have_the_right_reason():
    assert grading.make_record(PENGUINS, declined("bird_not_found")).behavior_ok
    wrong = grading.make_record(CAPITAL, declined("bird_not_found"))
    assert (wrong.expected, wrong.observed, wrong.behavior_ok) == (
        "decline:not_about_birds", "decline:bird_not_found", False)
    answered_anyway = grading.make_record(PENGUINS, answered())
    assert not answered_anyway.behavior_ok and not answered_anyway.needs_human_scores
    assert answered_anyway.right_bird is None


def test_wrongly_declining_an_in_scope_question_fails_and_is_not_graded():
    record = grading.make_record(SAKER, declined("detail_missing"))
    assert not record.behavior_ok and not record.needs_human_scores


def test_a_routed_question_naming_no_bird_should_ask():
    record = grading.make_record(OWLS, declined(router.SPECIES_NEEDED, route="status"))
    assert (record.expected, record.observed, record.behavior_ok) == ("ask", "ask", True)
    assert not record.needs_human_scores


def test_a_status_answer_without_an_assessment_needs_no_citation():
    record = grading.make_record(SWAN_GOOSE, answered("IUCN has no assessment.", [], route="status"))
    assert record.behavior_ok and record.route_ok and record.right_bird is None
    assert not record.needs_human_scores  # written from data: checked mechanically


def test_the_wrong_route_is_caught():
    assert not grading.make_record(SWAN_GOOSE, answered(route="descriptive")).route_ok


def test_a_pipeline_error_is_recorded_not_raised():
    record = grading.make_record(SAKER, None, "LLMError: overloaded")
    assert record.observed == "error" and not record.behavior_ok and not record.route_ok


def test_an_out_of_scope_reference_must_name_its_decline_reason():
    vague = Question("q", "out_of_scope", "?", (), (), True, reference_answer="Should decline.")
    with pytest.raises(ValueError, match="Decline \\(<reason>\\)"):
        grading.expected_behavior(vague)


def test_every_committed_question_has_a_known_expected_behavior():
    behaviors = {grading.expected_behavior(q) for q in evaluation.load_questions()}
    assert behaviors == {"answer", "ask", "decline:bird_not_found", "decline:not_about_birds"}


def test_a_run_writes_the_three_files_and_prefills_what_needs_no_grading(tmp_path):
    records = [grading.make_record(SAKER, answered()),
               grading.make_record(PENGUINS, declined("bird_not_found"))]
    grading.write_run(records, tmp_path)
    assert {p.name for p in tmp_path.iterdir()} == {"answers.jsonl", "sheet.md", "grades.csv"}
    rows = grading.load_grades(tmp_path / "grades.csv")
    assert [(r["id"], r["faithful"], r["behavior_ok"], r["right_bird"]) for r in rows] == [
        ("q03", "", "yes", "yes"), ("q16", "n/a", "yes", "")]
    sheet = (tmp_path / "sheet.md").read_text()
    assert "✍️ q03" in sheet and "✍️ q16" not in sheet and "**1 to grade**" in sheet
    assert "Text of source 1." in sheet  # the cited passage, to grade against


def test_load_grades_rejects_scores_off_the_scale(tmp_path):
    grading.write_run([grading.make_record(SAKER, answered())], tmp_path)
    path = tmp_path / "grades.csv"
    rows = list(csv.DictReader(path.open()))
    rows[0]["faithful"] = "3"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=grading.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="line 2 \\(q03\\): faithful is '3'"):
        grading.load_grades(path)


def row(id, faithful="", complete="", cited="", **checks):
    base = {"id": id, "split": "tune", "tags": "", "behavior_ok": "yes", "route_ok": "yes",
            "citations_valid": "yes", "right_bird": "yes"}
    return {**base, **checks, "faithful": faithful, "complete": complete, "cited": cited}


def test_summary_counts_checks_and_scores_only_fully_graded_answers():
    rows = [row("a", "2", "2", "2"), row("b", "1", "2", "0"),
            row("c", "2", "", "", right_bird="NO"),  # half graded: counted as ungraded
            row("d", "n/a", "n/a", "n/a", right_bird="")]  # nothing to grade
    s = grading.summarize_grades(rows)
    assert s.n == 4 and s.graded == 2 and s.ungraded == 1
    assert s.checks["right_bird"] == (2, 3) and s.checks["behavior_ok"] == (4, 4)
    assert s.means == {"faithful": 1.5, "complete": 2.0, "cited": 1.0}
    assert s.full_marks == {"faithful": 0.5, "complete": 1.0, "cited": 0.5}


def test_summary_of_an_ungraded_run():
    s = grading.summarize_grades([row("a")])
    assert s.graded == 0 and s.ungraded == 1 and s.means["faithful"] == 0.0
