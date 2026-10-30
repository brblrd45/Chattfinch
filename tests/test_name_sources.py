import httpx
import pandas as pd

from chattfinch import name_sources

SPECIES = pd.DataFrame({"species_code": ["a1", "b1", "c1"],
                        "common_name": ["A Bird", "B Bird", "C Bird"],
                        "scientific_name": ["Genus a", "Genus b", "Genus c"]})


def fake_fetchers(monkeypatch, calls, fail_wikipedia=False):
    def gbif(name):
        calls.append(("gbif", name))
        return {"match": None, "vernacular_names": []}

    def wikipedia(name, common_name):
        calls.append(("wikipedia", name))
        if fail_wikipedia:
            raise httpx.HTTPStatusError("429", request=None, response=None)
        return {"queried_name": name, "article": None}

    monkeypatch.setattr(name_sources, "fetch_gbif", gbif)
    monkeypatch.setattr(name_sources, "fetch_wikipedia", wikipedia)


def test_collect_caches_and_skips_finished_sources(monkeypatch, tmp_path):
    calls = []
    fake_fetchers(monkeypatch, calls)
    records = name_sources.collect(SPECIES, ("gbif", "wikipedia"), tmp_path)
    assert set(records) == {"a1", "b1", "c1"} and len(calls) == 6
    assert (tmp_path / "a1.json").exists()
    calls.clear()
    name_sources.collect(SPECIES, ("gbif", "wikipedia"), tmp_path)
    assert calls == []


def test_failed_source_is_retried_next_run(monkeypatch, tmp_path):
    calls = []
    fake_fetchers(monkeypatch, calls, fail_wikipedia=True)
    records = name_sources.collect(SPECIES, ("gbif", "wikipedia"), tmp_path)
    assert "gbif" in records["a1"] and "wikipedia" not in records["a1"]
    calls.clear()
    fake_fetchers(monkeypatch, calls)
    records = name_sources.collect(SPECIES, ("gbif", "wikipedia"), tmp_path)
    assert [c[0] for c in calls] == ["wikipedia"] * 3
    assert "wikipedia" in records["a1"]


def test_source_skipped_after_consecutive_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(name_sources, "MAX_CONSECUTIVE_FAILURES", 2)
    calls = []
    fake_fetchers(monkeypatch, calls, fail_wikipedia=True)
    name_sources.collect(SPECIES, ("gbif", "wikipedia"), tmp_path)
    assert [c[0] for c in calls].count("wikipedia") == 2
    assert [c[0] for c in calls].count("gbif") == 3
