import re

import pytest

from chattfinch import telegram_format as fmt
from chattfinch.rag import Answer, Source
from chattfinch.resolve import Match


def source(number, text=None, common="Saker Falcon", url=None):
    return Source(number=number, species_code="sakfal1", common_name=common,
                  scientific_name="Falco cherrug", section="Ecology", origin="wikipedia",
                  url=url or f"https://example.org/{number}", score=0.7,
                  text=text or f"Text of source {number}.")


def answer(text, numbers=(1, 2, 3), declined=False):
    return Answer(query="q", text=text, declined=declined, sources=[source(n) for n in numbers])


ALLOWED_TAGS = re.compile(r"</?(b|i)>|<a href=\"[^\"<>]*\">|</a>")


def assert_valid_telegram_html(message: str) -> None:
    """Only the tags Telegram accepts, every one closed, and no stray < or >."""
    assert len(message) <= fmt.MAX_MESSAGE
    remainder = ALLOWED_TAGS.sub("", message)
    assert "<" not in remainder and ">" not in remainder, remainder
    for tag in ("b", "i", "a"):
        assert len(re.findall(f"<{tag}[ >]", message)) == message.count(f"</{tag}>"), tag


def test_answer_links_citations_and_lists_cited_sources():
    (message,) = fmt.answer_messages(answer("Squirrels [1]. Pikas too [3]."))
    assert message.startswith('Squirrels <a href="https://example.org/1">[1]</a>. Pikas too '
                              '<a href="https://example.org/3">[3]</a>.')
    listing = message.split("<b>Sources</b>\n")[1].split("\n")
    assert listing == [
        '[1] <a href="https://example.org/1">Saker Falcon (Falco cherrug) — Ecology · '
        'Wikipedia (CC BY-SA 4.0)</a>',
        '[3] <a href="https://example.org/3">Saker Falcon (Falco cherrug) — Ecology · '
        'Wikipedia (CC BY-SA 4.0)</a>']  # [2] was retrieved but not cited
    assert_valid_telegram_html(message)


def test_grouped_and_adjacent_markers_each_become_a_link():
    (message,) = fmt.answer_messages(answer("Both [1][2] and grouped [1, 3]."))
    assert message.count('<a href="https://example.org/1">[1]</a>') == 2
    assert '<a href="https://example.org/2">[2]</a>' in message
    assert '<a href="https://example.org/3">[3]</a>' in message


def test_text_from_the_model_and_sources_is_escaped():
    hostile = Answer(query="q", text="Wings <b>bold</b> & <a href='x'>link</a> [1].", declined=False,
                     sources=[source(1, common="Bird <X> & \"Y\"", url='https://example.org/?a=1&b="2"')])
    (message,) = fmt.answer_messages(hostile)
    assert "Wings &lt;b&gt;bold&lt;/b&gt; &amp; &lt;a href='x'&gt;link&lt;/a&gt;" in message
    assert 'href="https://example.org/?a=1&amp;b=&quot;2&quot;"' in message
    assert "Bird &lt;X&gt; &amp; \"Y\"" in message
    assert_valid_telegram_html(message)


def test_invented_citation_is_not_linked_and_gets_a_caution():
    (message,) = fmt.answer_messages(answer("Real [1]. Invented [9]."))
    assert "Invented [9]." in message and "example.org/9" not in message
    assert "cites [9], which matches no source" in message
    assert_valid_telegram_html(message)


def test_declined_answer_shows_its_message_and_lists_no_sources():
    declined = answer("I couldn't find that bird in my sources <sic> & more.", declined=True)
    (message,) = fmt.answer_messages(declined)
    assert message == "I couldn't find that bird in my sources &lt;sic&gt; &amp; more."
    assert "Sources" not in message


def test_declined_answer_that_cites_the_passage_it_checked_is_linked_like_any_other():
    declined = answer("The sources give its diet [1] but not its lifespan.", declined=True)
    (message,) = fmt.answer_messages(declined)
    assert message.startswith('The sources give its diet <a href="https://example.org/1">[1]</a>')
    assert "<b>Sources</b>\n[1] " in message
    assert_valid_telegram_html(message)


def test_answer_without_citations_has_no_sources_block():
    (message,) = fmt.answer_messages(answer("Just a statement."))
    assert message == "Just a statement."


def test_sources_move_to_a_second_message_when_they_do_not_fit():
    long_answer = answer("Fact [1]. " + "Padding sentence. " * 50)  # fits alone, not with sources
    messages = fmt.answer_messages(long_answer, limit=1000)
    assert len(messages) == 2
    assert messages[0].startswith("Fact <a href") and messages[1].startswith("<b>Sources</b>")
    assert all(len(m) <= 1000 for m in messages)


def test_overlong_answer_is_cut_to_fit_one_message():
    huge = answer("Fact [1] & more. " * 600)  # ~10,000 characters
    messages = fmt.answer_messages(huge)
    assert all(len(m) <= fmt.MAX_MESSAGE for m in messages)
    assert messages[0].endswith("…")
    for message in messages:
        assert_valid_telegram_html(message)


def test_passages_list_cited_first_then_uncited_with_scores():
    (message,) = fmt.passage_messages(answer("B [3]. A [1]."))
    order = [message.index(f"Text of source {n}.") for n in (3, 1, 2)]
    assert order == sorted(order)
    assert message.index("Retrieved but not cited") < message.index("Text of source 2.")
    assert message.count("Similarity to the question: 0.70") == 3
    assert_valid_telegram_html(message)


def test_passages_split_across_messages_without_breaking_markup():
    sources = [source(n, text=f"Passage {n}: " + "feathers & <wings> " * 60) for n in range(1, 7)]
    result = Answer(query="q", text="All of them [1][2][3][4][5][6].", declined=False,
                    sources=sources)
    messages = fmt.passage_messages(result)
    assert len(messages) >= 2
    for message in messages:
        assert_valid_telegram_html(message)
    joined = "\n".join(messages)
    assert all(f"Passage {n}:" in joined for n in range(1, 7))
    assert "&lt;wings&gt;" in joined and "<wings>" not in joined


def test_single_passage_longer_than_a_message_is_truncated():
    result = Answer(query="q", text="It [1].", declined=False,
                    sources=[source(1, text="word & " * 2000)])
    (message,) = fmt.passage_messages(result)
    assert len(message) <= fmt.MAX_MESSAGE and message.endswith("…")
    assert_valid_telegram_html(message)


def test_passages_of_an_answer_with_no_sources_is_empty():
    assert fmt.passage_messages(Answer(query="q", text="x", declined=True, sources=[])) == []


def test_pack_joins_blocks_up_to_the_limit_and_never_splits_one():
    assert fmt.pack(["aaa", "bbb", "ccc"], limit=8) == ["aaa\n\nbbb", "ccc"]
    assert fmt.pack(["aaa", "bbb"], limit=100) == ["aaa\n\nbbb"]
    assert fmt.pack([], limit=10) == []
    with pytest.raises(ValueError, match="can't fit"):
        fmt.pack(["toolongblock"], limit=5)


def test_escape_to_fit_measures_the_escaped_length():
    fitted = fmt.escape_to_fit("&" * 100, limit=60)  # each & becomes 5 characters
    assert len(fitted) <= 60 and fitted.endswith("…") and "&amp;" in fitted
    assert fmt.escape_to_fit("short & sweet", limit=60) == "short &amp; sweet"


def test_plain_strips_markup_and_unescapes():
    (message,) = fmt.answer_messages(answer("Wings & <tail> [1]."))
    plain = fmt.plain(message)
    assert plain.startswith("Wings & <tail> [1].") and "<a href" not in plain and "<b>" not in plain
    assert "[1] Saker Falcon (Falco cherrug) — Ecology · Wikipedia (CC BY-SA 4.0)" in plain


# --- resolved names and choices --------------------------------------------------------

GOATSUCKER = Match("Common <goatsucker>", "common goatsucker", ("eurnig1",), False,
                   ("Eurasian Nightjar",))


def test_a_renamed_bird_is_named_above_the_answer():
    renamed = Answer(query="q", text="Moths [1].", declined=False, sources=[source(1)],
                     names=(GOATSUCKER,))
    (message,) = fmt.answer_messages(renamed)
    assert message.startswith("<i>Eurasian Nightjar (also: Common &lt;goatsucker&gt;)</i>\n\nMoths ")
    assert_valid_telegram_html(message)


def test_no_name_line_for_canonical_names_or_declines():
    assert fmt.names_line(answer("Moths [1].")) == ""
    declined = Answer(query="q", text="I couldn't find that bird.", declined=True, sources=[],
                      names=(GOATSUCKER,))
    assert fmt.answer_messages(declined) == ["I couldn't find that bird."]


def test_a_choice_is_just_the_escaped_question():
    choosing = Answer(query="q", text="Which bird do you mean by “Cat <Owl>”?", declined=False,
                      sources=[], choices=Match("Cat <Owl>", "cat owl", ("a", "b"), False))
    assert fmt.answer_messages(choosing) == ["Which bird do you mean by “Cat &lt;Owl&gt;”?"]


def test_looked_up_sources_show_no_similarity_score():
    status = Source(number=1, species_code="soclap1", common_name="Sociable Lapwing",
                    scientific_name="Vanellus gregarius", section="global assessment, 2019",
                    origin="iucn", url="https://www.iucnredlist.org/species/1/2", score=1.0,
                    text="Critically Endangered (CR), published 2019.")
    block = fmt.passage_block(status)
    assert "Similarity" not in block and "IUCN Red List" in block
    assert "Similarity to the question: 0.70" in fmt.passage_block(source(1))
