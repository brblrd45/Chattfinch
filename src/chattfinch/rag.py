"""End-to-end question answering: resolve names -> retrieve -> build prompt -> call LLM ->
answer + sources."""

from collections.abc import Callable
from dataclasses import dataclass

from chattfinch import config, llm, prompts, resolve, retrieval, router

Generate = Callable[[str, str], str]  # (system prompt, user message) -> reply text
Retrieve = Callable[..., list[dict]]  # retrieval.retrieve's signature
Resolver = Callable[[str], resolve.Resolution]  # resolve.resolve's signature

NOTHING_RETRIEVED = "I couldn't find anything in my sources for that question."
COULD_NOT_ANSWER = "I couldn't answer that from my sources."
CHOOSE = "Which bird do you mean by “{alias}”?"
# What the user reads when the model declines. Fixed wording for the two reasons where
# the retrieved passages are irrelevant, so no reply ever describes them. "May not be
# one of them" is deliberately hedged even with name resolution: a misspelled name
# ("Sakker Falcon") matches no alias, so it looks the same as a bird that isn't covered.
DECLINE_MESSAGES = {
    prompts.BIRD_NOT_FOUND: (f"I couldn't find that bird in my sources. I only cover birds "
                             f"recorded in {config.REGION_NAME}, so it may not be one of them."),
    prompts.NOT_ABOUT_BIRDS: (f"I can only answer questions about birds recorded in "
                              f"{config.REGION_NAME}."),
}


@dataclass(frozen=True)
class Source:
    """One retrieved chunk, numbered as the answer cites it ([1], [2], ...)."""
    number: int
    species_code: str
    common_name: str
    scientific_name: str
    section: str
    origin: str  # "wikipedia" or "ebird"
    url: str
    score: float  # cosine similarity to the question
    text: str  # the chunk body shown to the model


@dataclass(frozen=True)
class Answer:
    query: str
    text: str  # what to show the user: the answer, or the message explaining a decline
    declined: bool  # True when the sources didn't answer the question
    sources: list[Source]  # every chunk the model was shown, in citation order
    decline_reason: str | None = None  # one of prompts.DECLINE_REASONS, if the model gave one
    # The birds the question named, which retrieval was filtered to.
    names: tuple[resolve.Match, ...] = ()
    # Set when the question names a bird ambiguously: `text` asks which one is
    # meant, and nothing was retrieved or generated. Answer again with a choice.
    choices: resolve.Match | None = None
    route: str = router.DESCRIPTIVE  # which path answered: router.DESCRIPTIVE/STATUS/LIVE

    @property
    def needs_choice(self) -> bool:
        return self.choices is not None

    @property
    def renamed(self) -> list[tuple[str, str]]:
        """(name as asked, canonical name) for each bird asked about by another name."""
        return [m.renamed for m in self.names if m.renamed]

    @property
    def sources_are_relevant(self) -> bool:
        """False when the retrieved passages have nothing to do with the question
        (the bird wasn't found, or it wasn't about birds), so showing them would
        only confuse."""
        return self.decline_reason not in (prompts.BIRD_NOT_FOUND, prompts.NOT_ABOUT_BIRDS)


def _sources(hits: list[dict]) -> list[Source]:
    return [Source(number=prompts.source_number(position),
                   species_code=hit["metadata"]["species_code"],
                   common_name=hit["metadata"]["common_name"],
                   scientific_name=hit["metadata"]["scientific_name"],
                   section=hit["metadata"]["section"],
                   origin=hit["metadata"]["source"],
                   url=hit["metadata"]["url"],
                   score=hit["score"],
                   text=prompts.body(hit))
            for position, hit in enumerate(hits)]


def parse_reply(reply: str) -> tuple[str, bool, str | None]:
    """(text to show, declined, decline reason) from the model's reply.

    A reply starting with the NO_ANSWER marker is a decline. For "bird not found"
    and "not about birds" the text is our fixed message and anything the model wrote
    is dropped; for "detail missing" it is the model's sentence about what's missing.
    A decline with no recognizable reason keeps the model's words.
    """
    reply = reply.strip()
    if not reply.startswith(prompts.NO_ANSWER):
        return reply, False, None
    rest = reply[len(prompts.NO_ANSWER):].lstrip(" :.\t")
    first_line, _, remainder = rest.partition("\n")
    reason = next((r for r in prompts.DECLINE_REASONS
                   if first_line.strip().lower().rstrip(".:") == r), None)
    if reason is None:  # no reason given: all of it is the model's explanation
        return f"{COULD_NOT_ANSWER} {rest.strip()}".strip(), True, None
    if reason in DECLINE_MESSAGES:
        return DECLINE_MESSAGES[reason], True, reason
    return remainder.strip() or COULD_NOT_ANSWER, True, reason


def search_text(query: str, names: list[resolve.Match]) -> str:
    """The query to search with: each bird named by another name is replaced by its
    canonical name. Retrieval is already filtered to that species, so the typed name
    adds nothing but noise: a passage that happens to list it ("also known as the
    Eurasian black vulture") outranks the passage that answers the question."""
    for match in sorted((m for m in names if m.renamed and m.span), key=lambda m: m.span,
                        reverse=True):
        start, end = match.span
        query = query[:start] + match.common_names[0] + query[end:]
    return query


def search(query: str, k: int, species: tuple[str, ...] = (),
           retrieve: Retrieve = retrieval.retrieve) -> list[dict]:
    """The `k` best chunks for `query`, from `species` only if any are given.

    With several species, each gets an equal share of the k places (best first
    overall), so a comparison question sees every bird it names rather than
    whichever one matches best. If the filter finds nothing, search everything.
    """
    if not species:
        return retrieve(query, k=k)
    share = -(-k // len(species))  # ceiling division
    hits = [hit for code in species for hit in retrieve(query, k=share, species_filter=code)]
    if not hits:
        return retrieve(query, k=k)
    return sorted(hits, key=lambda hit: hit["score"], reverse=True)[:k]


def answer(query: str, k: int = config.RETRIEVAL_K, generate: Generate = llm.generate,
           retrieve: Retrieve = retrieval.retrieve, resolver: Resolver = resolve.resolve,
           choice: str | None = None, snapshot: Callable[[], dict | None] = router.load_snapshot,
           sightings: Callable[[str], list[dict]] = router.recent_sightings) -> Answer:
    """Answer `query` from the `k` best-matching chunks.

    Bird names in the query are resolved first, and retrieval is filtered to the
    species they name. If a name is ambiguous ("cat owl"), the returned Answer asks
    which bird is meant (`needs_choice`) and nothing is retrieved or generated;
    call again with `choice`, one of `answer.choices.species`. Only the first
    ambiguous name is asked about.

    Status questions ("Is the Saker Falcon endangered?") are answered from the IUCN
    snapshot and live ones ("seen recently?") from eBird, without retrieval or a
    model call (see router.py); `snapshot` and `sightings` supply that data.

    The returned sources are exactly what the model saw, numbered the way the
    answer cites them. If retrieval finds nothing, the model isn't called.
    Raises ValueError for an empty query or an invalid choice, and llm.LLMError if
    generation fails.
    """
    resolution = resolver(query)
    names = list(resolution.species_matches)
    pending = resolution.ambiguous
    if pending is not None and choice is None:
        return Answer(query=query, text=CHOOSE.format(alias=pending.alias), declined=False,
                      sources=[], names=tuple(names), choices=pending)
    if choice is not None:
        if pending is None:
            raise ValueError(f"Nothing to choose in {query!r}")
        names.append(pending.choose(choice))
    species = tuple(dict.fromkeys(m.species[0] for m in names))

    route = router.classify(query)
    if route != router.DESCRIPTIVE:
        reply = (router.status_reply(names, snapshot()) if route == router.STATUS
                 else router.live_reply(names, sightings))
        return Answer(query=query, text=reply.text, declined=reply.declined,
                      sources=[Source(**fields) for fields in reply.sources],
                      decline_reason=reply.reason, names=tuple(names), route=route)

    hits = search(search_text(query, names), k, species, retrieve)
    if not hits:
        return Answer(query=query, text=NOTHING_RETRIEVED, declined=True, sources=[],
                      names=tuple(names))
    renamed = [m.renamed for m in names if m.renamed]
    reply = generate(prompts.SYSTEM_PROMPT, prompts.build_user_message(query, hits, renamed))
    text, declined, reason = parse_reply(reply)
    return Answer(query=query, text=text, declined=declined, sources=_sources(hits),
                  decline_reason=reason, names=tuple(names))
