"""The Streamlit app, run headlessly with Streamlit's test harness. The index load
and the answer pipeline are replaced by fakes: no key, no index, no browser."""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from chattfinch import config, llm, prompts, rag, retrieval
from chattfinch.resolve import Match
from test_citations import source

APP = str(config.ROOT / "src" / "chattfinch" / "app.py")


@pytest.fixture
def pipeline(monkeypatch):
    """Fakes for the index load and rag.answer. `.result` is what the next question
    returns (an Answer, or an exception to raise); `.questions` records what was asked."""
    state = type("Pipeline", (), {})()
    state.questions, state.choices = [], []
    state.result = rag.Answer(query="q", text="It hunts ground squirrels [1] and pikas [3].",
                              declined=False, sources=[source(1), source(2), source(3)])

    def fake_answer(question, choice=None):
        state.questions.append(question)
        state.choices.append(choice)
        if callable(state.result):  # a function of the choice, for the choice tests
            return state.result(choice)
        if isinstance(state.result, Exception):
            raise state.result
        return state.result

    monkeypatch.setattr(retrieval, "warm_up", lambda: None)
    monkeypatch.setattr(rag, "answer", fake_answer)
    st.cache_data.clear()
    st.cache_resource.clear()
    yield state
    st.cache_data.clear()
    st.cache_resource.clear()


def run_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception, app.exception
    return app


def ask(app: AppTest, question: str) -> AppTest:
    app.text_input(key="question").set_value(question)
    next(b for b in app.button if b.label == "Ask").click()
    app.run()
    assert not app.exception, app.exception
    return app


def all_markdown(app: AppTest) -> str:
    return "\n".join(m.value for m in app.markdown)


def test_page_loads_with_a_question_box_and_no_answer(pipeline):
    app = run_app()
    assert app.title[0].value == "Ask a Birder"
    assert app.text_input(key="question").label == "Your question"
    assert config.REGION_NAME in app.caption[0].value
    assert pipeline.questions == [] and not app.expander and not app.error


def test_asking_shows_the_answer_with_linked_citations(pipeline):
    app = ask(run_app(), "  What does the Saker Falcon eat?  ")
    assert pipeline.questions == ["What does the Saker Falcon eat?"]  # stripped
    markdown = all_markdown(app)
    assert "ground squirrels [\\[1\\]](https://example.org/1)" in markdown
    assert "pikas [\\[3\\]](https://example.org/3)" in markdown


def test_cited_sources_are_listed_under_the_answer(pipeline):
    app = ask(run_app(), "What does it eat?")
    listing = next(m.value for m in app.markdown if m.value.startswith("**Sources**"))
    assert "**\\[1\\]**" in listing and "**\\[3\\]**" in listing
    assert "**\\[2\\]**" not in listing  # retrieved but not cited
    assert "Wikipedia (CC BY-SA 4.0)" in listing and "(https://example.org/1)" in listing


def test_sources_panel_shows_every_retrieved_passage(pipeline):
    app = ask(run_app(), "What does it eat?")
    (panel,) = app.expander
    assert "all 3 retrieved passages" in panel.label
    assert [t.value for t in panel.text] == [
        "Text of source 1.", "Text of source 3.", "Text of source 2."]  # cited first
    assert any("Retrieved but not cited" in m.value for m in panel.markdown)
    assert any("Similarity to the question: 0.70" in m.value for m in panel.markdown)


def test_declined_answer_shows_its_message(pipeline):
    pipeline.result = rag.Answer(query="q", text="The sources don't give its lifespan.",
                                 declined=True, sources=[source(1), source(2)],
                                 decline_reason=prompts.DETAIL_MISSING)
    app = ask(run_app(), "How long does it live?")
    assert app.info[0].value == "The sources don't give its lifespan."
    assert not any(m.value.startswith("**Sources**") for m in app.markdown)  # nothing cited
    assert app.expander[0].label == "Show the text of all 2 retrieved passages"


def test_bird_not_found_marks_the_passages_as_unrelated(pipeline):
    pipeline.result = rag.Answer(query="q", text=rag.DECLINE_MESSAGES[prompts.BIRD_NOT_FOUND],
                                 declined=True, sources=[source(1)],
                                 decline_reason=prompts.BIRD_NOT_FOUND)
    app = ask(run_app(), "Where does the tufted titmouse live?")
    assert "may not be one of them" in app.info[0].value
    assert app.expander[0].label.endswith("(unrelated to the question)")


def test_nothing_retrieved_shows_no_sources_panel(pipeline):
    pipeline.result = rag.Answer(query="q", text=rag.NOTHING_RETRIEVED, declined=True, sources=[])
    app = ask(run_app(), "Anything?")
    assert app.info and not app.expander


def test_invented_citation_gets_a_warning(pipeline):
    pipeline.result = rag.Answer(query="q", text="Real [1]. Invented [9].", declined=False,
                                 sources=[source(1)])
    app = ask(run_app(), "A question")
    assert "[9]" in app.warning[0].value and "caution" in app.warning[0].value


def test_model_failure_shows_the_error_message(pipeline):
    pipeline.result = llm.LLMError("No valid Anthropic credentials. Set ANTHROPIC_API_KEY in .env")
    app = ask(run_app(), "A question")
    assert "ANTHROPIC_API_KEY" in app.error[0].value
    assert not app.expander


def test_empty_question_asks_for_one_without_calling_the_model(pipeline):
    app = ask(run_app(), "   ")
    assert app.warning[0].value == "Type a question first."
    assert pipeline.questions == []


def test_repeating_a_question_does_not_call_the_model_again(pipeline):
    app = ask(run_app(), "Same question?")
    ask(app, "Same question?")
    ask(run_app(), "Same question?")  # a new browser session too
    assert pipeline.questions == ["Same question?"]


def test_example_button_asks_that_question(pipeline):
    app = run_app()
    example = next(b for b in app.button if b.label == "Where do Demoiselle Cranes spend the winter?")
    example.click()
    app.run()
    assert not app.exception
    assert pipeline.questions == ["Where do Demoiselle Cranes spend the winter?"]
    assert app.text_input(key="question").value == "Where do Demoiselle Cranes spend the winter?"
    assert app.expander


def test_missing_index_explains_how_to_build_it(pipeline, monkeypatch):
    def no_index():
        raise FileNotFoundError("No index at data/index; run scripts/build_index.py")
    monkeypatch.setattr(retrieval, "warm_up", no_index)
    app = run_app()
    assert "build_index.py" in app.error[0].value
    assert not app.text_input  # the page stops there


CAT_OWL = Match("Cat Owl", "cat owl", ("loeowl", "tawowl1"), False, ("Long-eared Owl", "Tawny Owl"))


def test_an_ambiguous_name_offers_a_button_per_bird_and_answers_the_choice(pipeline):
    def result(choice):
        if choice is None:
            return rag.Answer(query="q", text="Which bird do you mean by “Cat Owl”?",
                              declined=False, sources=[], choices=CAT_OWL)
        return rag.Answer(query="q", text="Yes [1].", declined=False, sources=[source(1)],
                          names=(CAT_OWL.choose(choice),))
    pipeline.result = result
    app = ask(run_app(), "Is the cat owl nocturnal?")
    assert app.info[0].value == "Which bird do you mean by “Cat Owl”?"
    next(b for b in app.button if b.label == "Tawny Owl").click()
    app.run()
    assert pipeline.choices == [None, "tawowl1"]
    assert any("Tawny Owl (also: Cat Owl)" in c.value for c in app.caption)
