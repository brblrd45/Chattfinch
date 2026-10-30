"""The answer pipeline with a fake retriever and a fake model: no key, no index."""

from dataclasses import replace

import pytest

from chattfinch import config, llm, prompts, rag
from chattfinch.resolve import Match, Resolution
from test_prompts import hit

HITS = [hit(), hit(common="Red-footed Falcon", scientific="Falco vespertinus",
                   section="Dietary habits", body="Nestlings eat beetles.", code="reffal1",
                   score=0.6)]


class FakeModel:
    """Records what it was asked and replies with a canned string."""

    def __init__(self, reply="The saker falcon hunts ground squirrels [1]."):
        self.reply, self.calls = reply, []

    def __call__(self, system, user):
        self.calls.append((system, user))
        return self.reply


def fake_retrieve(hits=HITS):
    calls = []

    def retrieve(query, k, species_filter=None):
        calls.append((query, k) if species_filter is None else (query, k, species_filter))
        return hits
    retrieve.calls = calls
    return retrieve


def test_answer_retrieves_builds_the_prompt_and_calls_the_model_once():
    model, retrieve = FakeModel(), fake_retrieve()
    result = rag.answer("What does the saker falcon eat?", generate=model, retrieve=retrieve)
    # "saker falcon" is resolved to its species code, and retrieval filtered to it.
    assert retrieve.calls == [("What does the saker falcon eat?", config.RETRIEVAL_K, "sakfal1")]
    (system, user), = model.calls
    assert system == prompts.SYSTEM_PROMPT
    assert user == prompts.build_user_message("What does the saker falcon eat?", HITS)
    assert result.query == "What does the saker falcon eat?"
    assert result.text == "The saker falcon hunts ground squirrels [1]."
    assert result.declined is False


def test_k_is_passed_to_retrieval():
    retrieve = fake_retrieve()
    rag.answer("q", k=3, generate=FakeModel(), retrieve=retrieve)
    assert retrieve.calls == [("q", 3)]


def test_sources_match_what_the_model_saw_number_for_number():
    model = FakeModel()
    result = rag.answer("q", generate=model, retrieve=fake_retrieve())
    user_message = model.calls[0][1]
    assert [s.number for s in result.sources] == [1, 2]
    for source in result.sources:
        # The block the model saw under this number names the same species and text.
        block = user_message.split(f'<source number="{source.number}" ')[1].split("</source>")[0]
        assert f'species="{source.common_name}"' in block and source.text in block
    first = result.sources[0]
    assert (first.species_code, first.scientific_name, first.section, first.origin) == (
        "sakfal1", "Falco cherrug", "Ecology", "wikipedia")
    assert first.url == "https://example.org/sakfal1" and first.score == 0.7
    assert first.text == "It hunts ground squirrels."  # body only, no header line


NOT_FOUND = rag.DECLINE_MESSAGES[prompts.BIRD_NOT_FOUND]
NOT_BIRDS = rag.DECLINE_MESSAGES[prompts.NOT_ABOUT_BIRDS]


@pytest.mark.parametrize("reply, expected_text, expected_reason", [
    ("NO_ANSWER bird_not_found", NOT_FOUND, "bird_not_found"),
    ("  NO_ANSWER bird_not_found  \n", NOT_FOUND, "bird_not_found"),
    ("NO_ANSWER: BIRD_NOT_FOUND.", NOT_FOUND, "bird_not_found"),  # tolerant of case and punctuation
    ("NO_ANSWER not_about_birds", NOT_BIRDS, "not_about_birds"),
    ("NO_ANSWER not_about_birds\nAstana is the capital.", NOT_BIRDS, "not_about_birds"),
    ("NO_ANSWER detail_missing\nThe sources describe its diet but not its lifespan.",
     "The sources describe its diet but not its lifespan.", "detail_missing"),
    ("NO_ANSWER detail_missing", rag.COULD_NOT_ANSWER, "detail_missing"),
    # No recognizable reason: keep the model's words behind a plain lead-in.
    ("NO_ANSWER\nNothing found.", "I couldn't answer that from my sources. Nothing found.", None),
    ("NO_ANSWER: nothing about penguins here.",
     "I couldn't answer that from my sources. nothing about penguins here.", None),
    ("NO_ANSWER", rag.COULD_NOT_ANSWER, None),
])
def test_decline_marker_and_reason_decide_the_message(reply, expected_text, expected_reason):
    result = rag.answer("q", generate=FakeModel(reply), retrieve=fake_retrieve())
    assert result.declined is True
    assert (result.text, result.decline_reason) == (expected_text, expected_reason)
    assert len(result.sources) == 2  # still returned, for the eval and for inspection


def test_bird_not_found_never_repeats_what_the_unrelated_sources_were_about():
    """Regression: asked about the tufted titmouse (not a Kazakhstan bird), the model
    replied that the sources "cover only sandgrouse species", which reads as if
    sandgrouse had something to do with the question."""
    reply = ("NO_ANSWER bird_not_found\nNone of the sources mention the tufted titmouse. They "
             "cover only sandgrouse species (pin-tailed, black-bellied and Pallas's sandgrouse).")
    result = rag.answer("Where does the tufted titmouse live?", generate=FakeModel(reply),
                        retrieve=fake_retrieve())
    assert result.text == NOT_FOUND
    assert "sandgrouse" not in result.text
    assert "Kazakhstan" in result.text and "may not be one of them" in result.text  # hedged


def test_sources_are_relevant_unless_the_bird_was_not_found_or_it_was_not_about_birds():
    relevant = lambda reply: rag.answer("q", generate=FakeModel(reply),
                                        retrieve=fake_retrieve()).sources_are_relevant
    assert relevant("It hunts squirrels [1].")
    assert relevant("NO_ANSWER detail_missing\nNothing on lifespan.")
    assert relevant("NO_ANSWER\nUnclear.")
    assert not relevant("NO_ANSWER bird_not_found")
    assert not relevant("NO_ANSWER not_about_birds")


def test_an_answer_has_no_decline_reason():
    result = rag.answer("q", generate=FakeModel(), retrieve=fake_retrieve())
    assert result.declined is False and result.decline_reason is None


def test_marker_only_counts_at_the_start_of_the_reply():
    reply = "It eats squirrels [1]. (I would have said NO_ANSWER otherwise.)"
    result = rag.answer("q", generate=FakeModel(reply), retrieve=fake_retrieve())
    assert result.declined is False and result.text == reply


def test_nothing_retrieved_declines_without_calling_the_model():
    model = FakeModel()
    result = rag.answer("q", generate=model, retrieve=fake_retrieve(hits=[]))
    assert model.calls == []
    assert result.declined is True and result.sources == []
    assert result.text == rag.NOTHING_RETRIEVED


def test_model_failure_propagates():
    def broken(system, user):
        raise llm.LLMError("rate-limited")
    with pytest.raises(llm.LLMError, match="rate-limited"):
        rag.answer("q", generate=broken, retrieve=fake_retrieve())


def test_empty_question_is_rejected_before_any_model_call():
    model = FakeModel()
    with pytest.raises(ValueError, match="Empty query"):
        rag.answer("   ", generate=model)  # real retrieval: it validates before searching
    assert model.calls == []


# --- name resolution in the pipeline ---------------------------------------------------

GOATSUCKER = Match("Common goatsucker", "common goatsucker", ("eurnig1",), False,
                   ("Eurasian Nightjar",))
CAT_OWL = Match("Cat Owl", "cat owl", ("loeowl", "tawowl1"), False, ("Long-eared Owl", "Tawny Owl"))
OWL = Match("Owl", "owl", ("loeowl", "tawowl1", "eueowl1"), True)


def resolver(*matches):
    return lambda query: Resolution(matches=matches)


def test_a_renamed_bird_filters_retrieval_and_is_explained_to_the_model():
    model, retrieve = FakeModel(), fake_retrieve()
    goatsucker = replace(GOATSUCKER, span=(14, 31))
    result = rag.answer("What does the common goatsucker eat?", generate=model, retrieve=retrieve,
                        resolver=resolver(goatsucker))
    # The search uses the canonical name; the model still sees the question as asked.
    assert retrieve.calls == [("What does the Eurasian Nightjar eat?", config.RETRIEVAL_K,
                               "eurnig1")]
    user = model.calls[0][1]
    assert "<question>\nWhat does the common goatsucker eat?\n</question>" in user
    assert '<names>\n"Common goatsucker" in the question means the Eurasian Nightjar.\n</names>' in user
    assert user.index("<names>") < user.index("<question>")
    assert result.names == (goatsucker,)
    assert result.renamed == [("Common goatsucker", "Eurasian Nightjar")]


def test_no_names_block_when_nothing_was_renamed():
    model = FakeModel()
    rag.answer("q", generate=model, retrieve=fake_retrieve(), resolver=resolver())
    assert "<names>" not in model.calls[0][1]


def test_a_group_name_does_not_filter():
    retrieve = fake_retrieve()
    rag.answer("What do owls eat?", generate=FakeModel(), retrieve=retrieve, resolver=resolver(OWL))
    assert retrieve.calls == [("What do owls eat?", config.RETRIEVAL_K)]


def test_an_ambiguous_name_asks_which_bird_without_retrieving_or_generating():
    model, retrieve = FakeModel(), fake_retrieve()
    result = rag.answer("Is the cat owl nocturnal?", generate=model, retrieve=retrieve,
                        resolver=resolver(CAT_OWL))
    assert result.needs_choice and result.choices == CAT_OWL
    assert result.text == "Which bird do you mean by “Cat Owl”?"
    assert not result.declined and result.sources == []
    assert retrieve.calls == [] and model.calls == []


def test_answering_with_a_choice_filters_to_the_chosen_bird():
    model, retrieve = FakeModel(), fake_retrieve()
    result = rag.answer("Is the cat owl nocturnal?", generate=model, retrieve=retrieve,
                        resolver=resolver(CAT_OWL), choice="tawowl1")
    assert not result.needs_choice
    assert retrieve.calls == [("Is the cat owl nocturnal?", config.RETRIEVAL_K, "tawowl1")]
    assert '"Cat Owl" in the question means the Tawny Owl.' in model.calls[0][1]


@pytest.mark.parametrize("matches, choice", [((CAT_OWL,), "perfal"), ((), "tawowl1")])
def test_an_invalid_choice_raises(matches, choice):
    with pytest.raises(ValueError):
        rag.answer("q", generate=FakeModel(), retrieve=fake_retrieve(), resolver=resolver(*matches),
                   choice=choice)


def test_search_shares_the_places_between_several_species():
    def retrieve(query, k, species_filter=None):
        calls.append((k, species_filter))
        return [hit(code=species_filter, score=score) for score in ([0.9, 0.8, 0.7]
                if species_filter == "a" else [0.85, 0.5, 0.4])][:k]
    calls = []
    hits = rag.search("q", 4, ("a", "b"), retrieve)
    assert calls == [(2, "a"), (2, "b")]
    assert [(h["metadata"]["species_code"], h["score"]) for h in hits] == [
        ("a", 0.9), ("b", 0.85), ("a", 0.8), ("b", 0.5)]


def test_search_falls_back_to_everything_when_the_filter_finds_nothing():
    calls = []

    def retrieve(query, k, species_filter=None):
        calls.append(species_filter)
        return [] if species_filter else HITS
    assert rag.search("q", 6, ("nosuch1",), retrieve) == HITS
    assert calls == ["nosuch1", None]


def test_search_text_swaps_each_renamed_bird_for_its_canonical_name():
    query = "Is the common goatsucker bigger than the Cat Owl or a Tawny Owl?"
    names = [replace(GOATSUCKER, span=(7, 24)),
             replace(CAT_OWL, span=(41, 48)).choose("tawowl1"),
             Match("Tawny Owl", "tawny owl", ("tawowl1",), False, ("Tawny Owl",), span=(54, 63))]
    assert rag.search_text(query, names) == (
        "Is the Eurasian Nightjar bigger than the Tawny Owl or a Tawny Owl?")
    assert rag.search_text(query, []) == query
    assert rag.search_text(query, [GOATSUCKER]) == query  # no span: left alone
