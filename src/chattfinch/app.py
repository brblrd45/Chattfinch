"""Streamlit demo: ask a question, read the answer, check it against its sources.

Run from the repo root:

    .venv/bin/streamlit run src/chattfinch/app.py

Needs the built index (scripts/build_index.py) and ANTHROPIC_API_KEY in .env.
"""

import streamlit as st

from chattfinch import citations, config, llm, rag, retrieval

EXAMPLES = [
    "What does the Saker Falcon eat?",
    "Where do Demoiselle Cranes spend the winter?",
    "Why has the Sociable Lapwing declined?",
]


@st.cache_resource(show_spinner="Loading the search index…")
def load_index() -> bool:
    """Load the embedding model and the vector store once per server process, not
    on every question or page interaction."""
    retrieval.warm_up()
    return True


@st.cache_data(show_spinner=False, max_entries=200)
def cached_answer(question: str, choice: str | None = None) -> rag.Answer:
    """Answers are cached by question text (and chosen bird), so repeating a
    question (or Streamlit re-running the page) doesn't pay for the same model call
    twice."""
    return rag.answer(question, choice=choice)


def use_example(example: str) -> None:
    st.session_state["question"] = example
    st.session_state["asked"] = example
    st.session_state["choice"] = None


def choose(code: str) -> None:
    st.session_state["choice"] = code


def show_source(source: rag.Source) -> None:
    score = ("" if source.origin in citations.LOOKED_UP
             else f"  \nSimilarity to the question: {source.score:.2f}")
    st.markdown(f"**\\[{source.number}\\]** [{citations.source_label(source)}]({source.url})"
                + score)
    st.text(source.text)


def show_answer(answer: rag.Answer) -> None:
    if answer.needs_choice:
        st.info(answer.text)
        for code, name in zip(answer.choices.species, answer.choices.common_names):
            st.button(name, key=f"choose-{code}", on_click=choose, args=(code,))
        return
    for asked, common in answer.renamed:
        st.caption(f"{common} (also: {asked})")
    if answer.declined:
        st.info(answer.text)  # already the full message explaining the decline
    else:
        st.markdown(citations.link_citations(answer))

    invalid = citations.invalid_citations(answer)
    if invalid:
        numbers = ", ".join(f"[{n}]" for n in invalid)
        st.warning(f"The answer cites {numbers}, which matches no source it was given. "
                   "Treat the answer with caution.")

    cited, uncited = citations.cited_sources(answer), citations.uncited_sources(answer)
    if cited:
        st.markdown("**Sources**\n\n" + citations.sources_markdown(cited))
    if answer.sources:
        note = "" if answer.sources_are_relevant else " (unrelated to the question)"
        with st.expander(f"Show the text of all {len(answer.sources)} retrieved passages{note}"):
            for source in cited:
                show_source(source)
            if uncited:
                st.markdown("---\n*Retrieved but not cited in the answer:*")
                for source in uncited:
                    show_source(source)


st.set_page_config(page_title="Ask a Birder", layout="centered")
st.title("Ask a Birder")
st.caption(f"Questions about the birds recorded in {config.REGION_NAME}, answered only from "
           "Wikipedia and eBird, with the sources shown.")

try:
    load_index()
except (FileNotFoundError, ValueError) as error:
    st.error(f"The search index isn't ready: {error}")
    st.stop()

with st.form("ask"):
    st.text_input("Your question", key="question",
                  placeholder="e.g. What does the Saker Falcon eat?")
    if st.form_submit_button("Ask", type="primary"):
        st.session_state["asked"] = st.session_state["question"]
        st.session_state["choice"] = None

st.caption("Or try an example:")
for column, example in zip(st.columns(len(EXAMPLES)), EXAMPLES):
    column.button(example, on_click=use_example, args=(example,), use_container_width=True)

asked = st.session_state.get("asked")
if asked is not None:
    if not asked.strip():
        st.warning("Type a question first.")
    else:
        try:
            with st.spinner("Searching the sources and writing an answer…"):
                answer = cached_answer(asked.strip(), st.session_state.get("choice"))
        except llm.LLMError as error:
            st.error(str(error))
        else:
            show_answer(answer)

st.divider()
st.caption("Article text is from Wikipedia, licensed CC BY-SA 4.0; each source links to the "
           "exact revision used. Species names and taxonomy are from eBird. Answers are "
           "written by an AI model from those passages and can be wrong: check the sources.")
