"""Query routing: the classifier, and the status and live answers built from data."""

import json

import httpx
import pytest

from chattfinch import config, evaluation, rag, resolve, router
from chattfinch.resolve import Match, Resolution

LAPWING = Match("Sociable Plover", "sociable plover", ("soclap1",), False, ("Sociable Lapwing",),
                scientific_names=("Vanellus gregarius",))
SWAN_GOOSE = Match("Swan Goose", "swan goose", ("swagoo1",), False, ("Swan Goose",),
                   scientific_names=("Anser cygnoides",))
SNAPSHOT = {"fetched": {"from": "2026-10-08", "to": "2026-10-11"}, "species": {
    "soclap1": {"scientific_name": "Vanellus gregarius", "iucn_name": "Vanellus gregarius",
                "category": "CR", "year_published": "2019",
                "assessment_url": "https://www.iucnredlist.org/species/1/2"},
    "swagoo1": {"scientific_name": "Anser cygnoides", "iucn_name": None, "category": None,
                "year_published": None, "assessment_url": None},
}}


@pytest.mark.parametrize("query, route", [
    ("Is the Saker Falcon endangered?", router.STATUS),
    ("What is the conservation status of the Sociable Lapwing?", router.STATUS),
    ("How threatened is the sociable lapwing", router.STATUS),
    ("What is the IUCN status of Otis tarda?", router.STATUS),
    ("Are Dalmatian Pelicans vulnerable?", router.STATUS),
    ("Is it on the Red List?", router.STATUS),
    ("Has the Sociable Lapwing been seen recently?", router.LIVE),
    ("Any recent sightings of Demoiselle Cranes?", router.LIVE),
    ("Where can I see a Great Bustard in Kazakhstan now?", router.LIVE),
    ("When was the last sighting of a Pallas's Sandgrouse?", router.LIVE),
    ("Have Black Larks been spotted lately?", router.LIVE),
    # Reasons and threats are explained by the corpus, not by a category.
    ("Why is the Eurasian Black Vulture threatened?", router.DESCRIPTIVE),
    ("What are the main threats to the Great Bustard?", router.DESCRIPTIVE),
    ("Is the Sociable Lapwing endangered because of hunting?", router.DESCRIPTIVE),
    ("Why has the Sociable Lapwing declined?", router.DESCRIPTIVE),
    # Range and abundance aren't live or Red List questions.
    ("Where can I see a Great Bustard?", router.DESCRIPTIVE),
    ("Is the Black Lark rare?", router.DESCRIPTIVE),
    ("What does the Saker Falcon eat?", router.DESCRIPTIVE),
])
def test_classify(query, route):
    assert router.classify(query) == route


def test_every_eval_question_takes_its_expected_route():
    """Each question names the path that should answer it; a descriptive question
    diverted to a lookup, or a lookup for the wrong bird, fails here."""
    results = evaluation.evaluate_routing(evaluation.load_questions(), router.classify,
                                          resolve.resolve)
    wrong = [(r.question.id, r.question.route, r.route, r.species) for r in results
             if not (r.route_ok and r.species_ok)]
    assert wrong == []


# --- status ------------------------------------------------------------------------------

def test_status_gives_the_category_year_and_source():
    reply = router.status_reply([LAPWING], SNAPSHOT)
    assert not reply.declined
    assert reply.text == ("The Sociable Lapwing is listed as Critically Endangered (CR) on the "
                          "IUCN Red List, in its global assessment published in 2019 [1].\n"
                          "(Red List data retrieved 2026-10-11.)")
    (source,) = reply.sources
    assert source["origin"] == "iucn" and source["url"] == "https://www.iucnredlist.org/species/1/2"
    rag.Source(**source)  # the fields rag needs


def test_status_for_a_species_without_an_assessment_says_so_without_guessing():
    reply = router.status_reply([SWAN_GOOSE], SNAPSHOT)
    assert reply.text == ("IUCN has no global Red List assessment for the Swan Goose under the "
                          "names I checked (Anser cygnoides).")
    assert reply.sources == [] and not reply.declined


def test_status_for_several_species_numbers_the_sources_in_order():
    reply = router.status_reply([SWAN_GOOSE, LAPWING, LAPWING], SNAPSHOT)
    assert reply.text.count("[1]") == 1 and "[2]" not in reply.text
    assert [s["species_code"] for s in reply.sources] == ["soclap1"]


def test_status_without_a_snapshot_declines_plainly():
    reply = router.status_reply([LAPWING], None)
    assert reply.declined and reply.reason == router.STATUS_UNAVAILABLE
    assert reply.text == router.NO_SNAPSHOT


@pytest.mark.parametrize("make_reply", [lambda: router.status_reply([], SNAPSHOT),
                                        lambda: router.live_reply([], lambda code: [])])
def test_a_routed_question_without_a_species_asks_for_one(make_reply):
    reply = make_reply()
    assert reply.declined and reply.reason == router.SPECIES_NEEDED
    assert "which bird you mean" in reply.text


def test_load_snapshot(tmp_path):
    assert router.load_snapshot(tmp_path / "missing.json") is None
    path = tmp_path / "iucn_snapshot.json"
    path.write_text(json.dumps(SNAPSHOT))
    assert router.load_snapshot(path) == SNAPSHOT


# --- live --------------------------------------------------------------------------------

def sighting(when, place, count=None, private=False):
    return {"species_code": "soclap1", "common_name": "Sociable Lapwing",
            "scientific_name": "Vanellus gregarius", "location": place,
            "location_private": private, "observed_at": when, "count": count, "lat": 0, "lng": 0}


def test_live_lists_the_most_recent_sightings_and_hides_private_places(monkeypatch):
    monkeypatch.setattr(config, "RECENT_SIGHTINGS_SHOWN", 2)
    observations = [sighting("2026-10-12 08:30", "Tengiz", 14), sighting("2026-10-05", "Home", 1,
                                                                         private=True),
                    sighting("2026-09-30", "Korgalzhyn")]
    reply = router.live_reply([LAPWING], lambda code: observations)
    assert reply.text == ("eBird has recent reports of the Sociable Lapwing from 3 locations in "
                          "Kazakhstan in the last 30 days [1]. Most recent first:\n"
                          "- 2026-10-12 08:30, Tengiz (14 birds)\n"
                          "- 2026-10-05, a private location (1 bird)")
    (source,) = reply.sources
    assert source["url"] == "https://ebird.org/species/soclap1/KZ"
    assert source["origin"] == "ebird_live" and "Korgalzhyn" in source["text"]
    assert "Home" not in reply.text + source["text"]


def test_no_sightings_is_not_reported_as_absence():
    reply = router.live_reply([LAPWING], lambda code: [])
    assert "no public eBird reports of the Sociable Lapwing" in reply.text
    assert "doesn't mean the bird isn't there" in reply.text
    assert "withholds reports of some sensitive species" in reply.text
    assert not reply.declined and reply.sources[0]["scientific_name"] == "Vanellus gregarius"


@pytest.mark.parametrize("failure, logged", [
    (httpx.ConnectTimeout("slow"), "ConnectTimeout (slow)"),
    (RuntimeError("EBIRD_API_KEY is not set"), "RuntimeError (EBIRD_API_KEY is not set)"),
    (httpx.HTTPStatusError("refused", request=httpx.Request("GET", "https://x"),
                           response=httpx.Response(403)), "HTTPStatusError (HTTP 403)"),
])
def test_an_ebird_failure_gets_a_plain_message_and_logs_the_cause(failure, logged, caplog):
    def sightings(code):
        raise failure
    with caplog.at_level("WARNING", logger="chattfinch.router"):
        reply = router.live_reply([LAPWING], sightings)
    assert reply.declined and reply.reason == router.LIVE_UNAVAILABLE
    assert reply.text == router.NO_EBIRD
    assert f"eBird lookup failed for soclap1: {logged}" in caplog.text


# --- in the pipeline ---------------------------------------------------------------------

def never(*args, **kwargs):
    raise AssertionError("should not be called")


def test_routed_questions_skip_retrieval_and_the_model():
    status = rag.answer("Is the Sociable Plover endangered?", generate=never, retrieve=never,
                        resolver=lambda q: Resolution(matches=(LAPWING,)),
                        snapshot=lambda: SNAPSHOT, sightings=never)
    assert status.route == router.STATUS and status.sources[0].origin == "iucn"
    assert status.renamed == [("Sociable Plover", "Sociable Lapwing")]
    live = rag.answer("Has the Sociable Plover been seen recently?", generate=never,
                      retrieve=never,
                      resolver=lambda q: Resolution(matches=(LAPWING,)),
                      snapshot=never, sightings=lambda code: [])
    assert live.route == router.LIVE and live.sources[0].origin == "ebird_live"


def test_descriptive_questions_are_marked_as_such():
    answer = rag.answer("q", generate=lambda s, u: "Text [1].",
                        retrieve=lambda query, k, species_filter=None: [], resolver=lambda q: Resolution(matches=()))
    assert answer.route == router.DESCRIPTIVE
