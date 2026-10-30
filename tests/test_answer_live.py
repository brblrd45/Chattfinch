"""The eval set's questions end to end, against the real index and the real model.

Skipped by default because every question is a paid API call. Run them with:

    .venv/bin/pytest tests/test_answer_live.py --live -v

They need the built index (scripts/build_index.py) and Anthropic credentials
(ANTHROPIC_API_KEY in .env). The checks are the mechanical part of the plan's
"faithfulness spot-check": an in-scope answer isn't declined, every citation
points at a source that was really shown, and at least one cited source is about
the right species; an out-of-scope question is declined. Whether each sentence is
actually supported by its source still needs a human read: use scripts/ask.py.

Routed questions (route "status" or "live") make no model call: status answers read
the private IUCN snapshot (skipped if it isn't on this machine) and live answers call
eBird (needs EBIRD_API_KEY). Their content isn't fixed (sightings change daily), so
the checks are about form: the right path, a citation to the right source for the
right bird, or a request to name a bird when the question names none.
"""

import os

import pytest

from chattfinch import citations, config, evaluation, rag, router
from chattfinch.clients import ebird_client

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not (config.INDEX_DIR / "chroma.sqlite3").exists(),
                       reason="index not built; run scripts/build_index.py"),
    pytest.mark.skipif(not (os.getenv("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_AUTH_TOKEN")),
                       reason="no Anthropic credentials; set ANTHROPIC_API_KEY in .env"),
]

# The questions live in eval/questions.jsonl, shared with scripts/eval_retrieval.py.
QUESTIONS = evaluation.load_questions()
DESCRIPTIVE = [q for q in QUESTIONS if q.route == router.DESCRIPTIVE]
IN_SCOPE = [q for q in DESCRIPTIVE if q.in_scope]
OUT_OF_SCOPE = [q for q in DESCRIPTIVE if not q.in_scope]
ROUTED = [q for q in QUESTIONS if q.route != router.DESCRIPTIVE]
ORIGIN = {router.STATUS: "iucn", router.LIVE: "ebird_live"}


@pytest.mark.parametrize("q", IN_SCOPE, ids=[f"{q.id} {q.type}: {q.question}" for q in IN_SCOPE])
def test_in_scope_question_is_answered_from_the_right_species(q):
    result = rag.answer(q.question)
    assert not result.declined, result.text
    assert citations.cited_numbers(result.text), f"no citations in: {result.text}"
    invalid = citations.invalid_citations(result)
    assert not invalid, f"cites sources it wasn't shown: {invalid}"
    cited_species = {source.species_code for source in citations.cited_sources(result)}
    assert cited_species & set(q.expected_species), (
        f"expected a citation to {q.expected_species}, got {sorted(cited_species)}: {result.text}")


@pytest.mark.parametrize("q", OUT_OF_SCOPE, ids=[f"{q.id}: {q.question}" for q in OUT_OF_SCOPE])
def test_out_of_scope_question_is_declined(q):
    result = rag.answer(q.question)
    assert result.declined, f"should decline ({q.note}) but answered: {result.text}"


@pytest.fixture
def real_ebird(monkeypatch):
    """conftest.py swaps in eBird clients with a dummy key (which eBird refuses with
    403), so no unit test can use the real key. These tests are about the real API:
    let the live client be built from EBIRD_API_KEY as the bot builds it."""
    monkeypatch.setattr(ebird_client, "_live_client", None)


@pytest.mark.parametrize("q", ROUTED, ids=[f"{q.id} {q.route}: {q.question}" for q in ROUTED])
def test_routed_question_takes_its_path_and_cites_the_right_lookup(q, real_ebird):
    if q.route == router.STATUS and router.load_snapshot() is None:
        pytest.skip("no IUCN snapshot in data/private/; run scripts/build_dumps.py")
    if q.route == router.LIVE and not config.EBIRD_API_KEY:
        pytest.skip("no eBird key; set EBIRD_API_KEY in .env")
    result = rag.answer(q.question)
    assert result.route == q.route, f"took the {result.route} path"
    if q.expect_declined:  # names no covered bird: should ask for one
        assert result.declined and result.decline_reason == router.SPECIES_NEEDED, result.text
        return
    assert not result.declined, f"{result.decline_reason}: {result.text}"
    assert not citations.invalid_citations(result)
    cited = {(s.species_code, s.origin) for s in citations.cited_sources(result)}
    looked_up = {(code, ORIGIN[q.route]) for code in q.expected_species}
    if q.route == router.STATUS and not cited:
        # A species without an IUCN assessment: no source, and the text must say so.
        assert "no global Red List assessment" in result.text, result.text
    else:
        assert cited == looked_up, f"cited {sorted(cited)}: {result.text}"
