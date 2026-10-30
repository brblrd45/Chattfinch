"""Name resolution: which birds a question names, found before retrieval.

Embeddings alone don't resolve aliases: "what does the common goatsucker eat" finds
nothing about nightjars, although the nightjar's profile lists the name. So the
question is scanned against the alias index first (data/dumps/aliases.csv), and
retrieval is then filtered to the species it names.

The question is normalized exactly as the aliases were (aliases.normalize), then
scanned left to right taking the longest alias at each position, so "eurasian eagle
owl" is one bird, not "eagle" + "owl", and "hedge sparrow" (the Dunnock) beats the
group name "sparrow".

Each alias found is one of:
  species    one species: retrieval is filtered to it
  ambiguous  2-4 species and not a group name ("cat owl"): the user is asked which
  group      a group name ("warbler", "eagle"), or more than 4 species: a topic,
             not a bird to resolve, so it doesn't filter anything

One-word aliases that are also ordinary English are reviewed by hand in
data/manual/everyday_words.csv:
  never  junk, never a match ("little" is a redirect to Little Tern)
  alone  a real bird name used only when the question names no other bird, so
         "how swift is the peregrine falcon" is about the Peregrine alone, while
         "where does the merlin breed" is still about the Merlin
"""

import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import pandas as pd

from chattfinch import aliases, config

ALIASES_PATH = config.DUMPS_DIR / "aliases.csv"
EVERYDAY_WORDS_PATH = config.MANUAL_DIR / "everyday_words.csv"
MAX_CANDIDATES = 4  # more than this and an alias is treated as a group: no wall of buttons
RULES = ("never", "alone")

SPECIES, AMBIGUOUS, GROUP = "species", "ambiguous", "group"
# A word of the question as typed: letters and digits, with internal apostrophes
# ("Pallas's"). Hyphens and other punctuation separate words, as in normalize().
_WORD = re.compile(r"[^\W_]+(?:['’][^\W_]+)*")
# Whose spelling to show for an alias: Wikipedia redirects are the messiest
# ("The Black-necked grebe"), so they come last.
_DISPLAY_PRIORITY = ("ebird", "iucn", "gbif", "scientific", "manual", "wikipedia", "group")


def _display_form(rows: pd.DataFrame) -> str:
    for source in _DISPLAY_PRIORITY:
        spelled = rows.loc[rows["sources"].str.split("|").map(lambda s: source in s), "alias"]
        if len(spelled):
            return spelled.iloc[0]
    return rows["alias"].iloc[0]


@dataclass(frozen=True)
class Match:
    """One alias found in the question."""
    alias: str  # as written in the alias table, e.g. "Common goatsucker"
    normalized: str
    species: tuple[str, ...]  # candidate species codes
    is_group: bool  # a group name like "warbler"
    common_names: tuple[str, ...] = ()  # the candidates' canonical names, same order
    span: tuple[int, int] | None = None  # where the question says it: query[start:end]
    scientific_names: tuple[str, ...] = ()  # same order as species

    @property
    def kind(self) -> str:
        if len(self.species) == 1:
            return SPECIES
        if self.is_group or len(self.species) > MAX_CANDIDATES:
            return GROUP
        return AMBIGUOUS

    def choose(self, code: str) -> "Match":
        """This match narrowed to one of its candidates (the user's choice)."""
        if code not in self.species:
            raise ValueError(f"{code!r} is not a candidate for {self.alias!r}")
        names = dict(zip(self.species, self.common_names))
        scientific = dict(zip(self.species, self.scientific_names))
        return replace(self, species=(code,), is_group=False,
                       common_names=(names[code],) if code in names else (),
                       scientific_names=(scientific[code],) if code in scientific else ())

    @property
    def renamed(self) -> tuple[str, str] | None:
        """(name as asked, canonical name) for a species asked about by another
        name ("Common goatsucker" -> "Eurasian Nightjar"); None otherwise."""
        if self.kind != SPECIES or not self.common_names:
            return None
        common = self.common_names[0]
        if aliases.normalize(common) == self.normalized:
            return None
        return self.alias, common


@dataclass(frozen=True)
class Resolution:
    matches: tuple[Match, ...]

    @property
    def species_matches(self) -> tuple[Match, ...]:
        return tuple(m for m in self.matches if m.kind == SPECIES)

    @property
    def species(self) -> tuple[str, ...]:
        """Species codes to filter retrieval to, in the order named, without repeats."""
        return tuple(dict.fromkeys(m.species[0] for m in self.species_matches))

    @property
    def ambiguous(self) -> Match | None:
        """The first alias the user has to disambiguate, if any."""
        return next((m for m in self.matches if m.kind == AMBIGUOUS), None)

    @property
    def groups(self) -> tuple[Match, ...]:
        return tuple(m for m in self.matches if m.kind == GROUP)


class NameIndex:
    """The alias table, the everyday-word rules and the species' own names."""

    def __init__(self, table: pd.DataFrame, everyday: pd.DataFrame, species: pd.DataFrame):
        self.names = {row.species_code: (row.common_name, row.scientific_name)
                      for row in species.itertuples()}
        self.entries: dict[str, Match] = {}
        for key, rows in table.groupby("normalized"):
            codes = tuple(sorted(set(rows["species_code"])))
            self.entries[key] = Match(alias=_display_form(rows), normalized=key, species=codes,
                                      is_group=bool((rows["kind"] == "group").any()),
                                      common_names=tuple(self.names[c][0] for c in codes),
                                      scientific_names=tuple(self.names[c][1] for c in codes))
        self.max_words = max(len(key.split()) for key in self.entries)
        bad = set(everyday["rule"]) - set(RULES)
        if bad:
            raise ValueError(f"{EVERYDAY_WORDS_PATH.name}: unknown rules {sorted(bad)}")
        self.never = frozenset(everyday.loc[everyday["rule"] == "never", "word"])
        self.alone = frozenset(everyday.loc[everyday["rule"] == "alone", "word"])

    def resolve(self, query: str) -> Resolution:
        # Normalized words with the span of the typed word each came from, so a match
        # can say where it is in the question.
        words = [(word, typed.span()) for typed in _WORD.finditer(query)
                 for word in aliases.normalize(typed.group()).split()]
        found, weak = [], []  # weak: "alone" words, kept only if nothing else matches
        i = 0
        while i < len(words):
            for n in range(min(self.max_words, len(words) - i), 0, -1):
                key = " ".join(word for word, _ in words[i:i + n])
                if key not in self.entries or (n == 1 and key in self.never):
                    continue
                match = replace(self.entries[key], span=(words[i][1][0], words[i + n - 1][1][1]))
                (weak if n == 1 and key in self.alone else found).append(match)
                i += n
                break
            else:
                i += 1
        return Resolution(matches=tuple(found or weak))


@lru_cache(maxsize=1)
def default_index(aliases_path: Path = ALIASES_PATH,
                  everyday_path: Path = EVERYDAY_WORDS_PATH,
                  species_path: Path = config.SPECIES_LIST_PATH) -> NameIndex:
    return NameIndex(pd.read_csv(aliases_path), pd.read_csv(everyday_path),
                     pd.read_csv(species_path))


def resolve(query: str) -> Resolution:
    """The birds `query` names, using the committed alias table."""
    return default_index().resolve(query)
