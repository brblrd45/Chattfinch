"""Split corpus documents into chunks sized for the embedding model.

Every chunk's embedded text is a one-line header naming the species and section,
then the body:

    Saker Falcon (Falco cherrug) — Behaviour > Feeding
    The saker falcon hunts ground squirrels, ...

The header is what lets "diet of the saker falcon" match a feeding paragraph that
never repeats the bird's name. It counts toward the token budget.

Bodies are packed from whole sentences. A sentence over the budget (profile alias
lists, long citations) is split at semicolons, then at words. Consecutive chunks
of one section share their boundary sentence when it's short, so a fact that
straddles a split is whole in at least one chunk.

Tokens are counted with the embedding model's own tokenizer, passed in as
`count_tokens`, so this module needs no model.
"""

import re
from collections.abc import Callable
from dataclasses import asdict, dataclass

SPECIAL_TOKENS = 2  # [CLS] and [SEP], which the model adds to every input
# A sentence ends at . ! ? before a capital letter, quote or bracket. Digits are
# deliberately excluded: "c. 1758" and "ca. 25 cm" are mid-sentence.
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z"“(\[])')
_SEMICOLON = re.compile(r"(?<=;)\s+")

CountTokens = Callable[[str], int]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    species_code: str
    common_name: str
    scientific_name: str
    source: str
    section: str
    url: str
    chunk_index: int  # position within the section
    text: str  # header + body: what gets embedded, retrieved and shown to the LLM
    n_tokens: int  # of `text`, including the two special tokens

    def to_dict(self) -> dict:
        return asdict(self)


def header(document: dict, section: dict) -> str:
    return (f"{document['common_name']} ({document['scientific_name']}) — "
            f"{section['section']}")


def _split_oversized(text: str, budget: int, count: CountTokens) -> list[str]:
    """Pieces of `text` that each fit `budget`: split at semicolons, then words."""
    pieces = []
    for part in _SEMICOLON.split(text):
        if count(part) <= budget:
            pieces.append(part)
            continue
        current: list[str] = []
        for word in part.split():
            if current and count(" ".join([*current, word])) > budget:
                pieces.append(" ".join(current))
                current = []
            current.append(word)
        if current:
            pieces.append(" ".join(current))
    # Re-join neighbours that fit together, so splits don't leave crumbs.
    merged: list[str] = []
    for piece in pieces:
        if merged and count(f"{merged[-1]} {piece}") <= budget:
            merged[-1] = f"{merged[-1]} {piece}"
        else:
            merged.append(piece)
    return merged


def _units(body: str, budget: int, count: CountTokens) -> list[tuple[str, int]]:
    """(text, tokens) for each sentence of `body`, every one within `budget`."""
    units = []
    for paragraph in re.split(r"\n+", body):
        for sentence in _SENTENCE_END.split(paragraph.strip()):
            sentence = sentence.strip()
            if not sentence:
                continue
            for piece in ([sentence] if count(sentence) <= budget
                          else _split_oversized(sentence, budget, count)):
                units.append((piece, count(piece)))
    return units


def pack(body: str, budget: int, overlap: int, count: CountTokens) -> list[str]:
    """Bodies of at most `budget` tokens each, packed from whole sentences."""
    chunks: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    used = 0
    for unit in _units(body, budget, count):
        if current and used + unit[1] > budget:
            chunks.append(current)
            last = current[-1]
            carry = [last] if last[1] <= overlap and last[1] + unit[1] <= budget else []
            current, used = carry, sum(n for _, n in carry)
        current.append(unit)
        used += unit[1]
    if current:
        chunks.append(current)
    return [" ".join(text for text, _ in chunk) for chunk in chunks]


def chunk_document(document: dict, count: CountTokens, max_tokens: int,
                   overlap_tokens: int) -> list[Chunk]:
    """All chunks of one corpus document, in section order. `max_tokens` is the cap
    on a chunk's embedded text including the two special tokens."""
    chunks = []
    for section in document["sections"]:
        head = header(document, section)
        budget = max_tokens - SPECIAL_TOKENS - count(head + "\n")
        if budget < 20:
            raise ValueError(f"Header leaves no room for text ({budget} tokens): {head!r}")
        for i, body in enumerate(pack(section["text"], budget, overlap_tokens, count)):
            text = f"{head}\n{body}"
            chunks.append(Chunk(
                chunk_id=f"{document['species_code']}:{section['source']}:{len(chunks)}",
                species_code=document["species_code"],
                common_name=document["common_name"],
                scientific_name=document["scientific_name"],
                source=section["source"],
                section=section["section"],
                url=section["url"],
                chunk_index=i,
                text=text,
                n_tokens=count(text) + SPECIAL_TOKENS,
            ))
    return chunks


def chunk_corpus(documents: list[dict], count: CountTokens, max_tokens: int,
                 overlap_tokens: int) -> list[Chunk]:
    return [chunk for document in documents
            for chunk in chunk_document(document, count, max_tokens, overlap_tokens)]
