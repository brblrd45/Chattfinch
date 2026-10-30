import httpx
import pandas as pd
import pytest

from chattfinch import species_list
from chattfinch.clients import ebird_client
from conftest import load_fixture

BASE = ebird_client.BASE_URL


@pytest.fixture
def ebird_kz(respx_mock):
    respx_mock.get(f"{BASE}/product/spplist/KZ").mock(
        return_value=httpx.Response(200, json=load_fixture("ebird/spplist_kz.json")))
    respx_mock.get(f"{BASE}/ref/taxonomy/ebird").mock(
        return_value=httpx.Response(200, json=load_fixture("ebird/taxonomy.json")))


def notable(*names: str) -> pd.DataFrame:
    return pd.DataFrame({"common_name": names, "scientific_name": names})


def test_keeps_all_species_and_drops_hybrids(ebird_kz):
    result = species_list.build_species_list("KZ", notable("Vanellus gregarius"))
    assert list(result.columns) == species_list.COLUMNS
    assert len(result) == 5  # fixture: 5 species + 1 hybrid
    assert not result["common_name"].str.contains("hybrid").any()
    assert result["taxon_order"].is_monotonic_increasing
    assert result.set_index("species_code")["is_notable"].to_dict() == {
        "gragoo": False, "eurnig1": False, "soclap1": True, "demcra1": False, "palsan1": False}
    assert result["mean_frequency"].isna().all()


def test_notable_species_missing_from_region_raises(ebird_kz):
    with pytest.raises(ValueError, match="Vanellus gregarious"):
        species_list.build_species_list("KZ", notable("Vanellus gregarious"))


def test_frequency_ranking_keeps_top_n_plus_notable(ebird_kz, monkeypatch):
    frequencies = pd.DataFrame({
        "common_name": ["Graylag Goose", "Eurasian Nightjar", "Demoiselle Crane",
                        "Pallas's Sandgrouse", "Sociable Lapwing"],
        "mean_frequency": [0.30, 0.20, 0.10, 0.05, 0.01]})
    monkeypatch.setattr(species_list, "load_frequencies", lambda path: frequencies)
    result = species_list.build_species_list("KZ", notable("Vanellus gregarius"),
                                             frequency_file="unused", target_count=2)
    assert set(result["species_code"]) == {"gragoo", "eurnig1", "soclap1"}


def test_shipped_notable_list_is_well_formed():
    frame = species_list.load_notable()
    assert list(frame.columns) == ["common_name", "scientific_name"]
    assert frame["scientific_name"].is_unique and frame.notna().all().all()


# --- The frozen list ---------------------------------------------------------------

def frozen_csv(tmp_path, rows=None):
    """A frozen species list on disk (default: what the ebird_kz fixtures would build)."""
    rows = rows or [
        ("gragoo", "Graylag Goose", "Anser anser", "Anseriformes", "Anatidae", "Ducks", 275.0),
        ("soclap1", "Sociable Lapwing", "Vanellus gregarius", "Charadriiformes", "Charadriidae", "Plovers", 5894.0)]
    path = tmp_path / "species_list.csv"
    pd.DataFrame(rows, columns=["species_code", "common_name", "scientific_name", "order", "family",
                                "family_common_name", "taxon_order"]).assign(
        is_notable=False, mean_frequency=pd.NA).to_csv(path, index=False)
    return path


def test_load_frozen_never_calls_ebird(respx_mock, tmp_path):
    # respx_mock fails any request that isn't mocked, and none are mocked here.
    species = species_list.load_frozen(notable("Vanellus gregarius"), frozen_csv(tmp_path))
    assert list(species.columns) == species_list.COLUMNS
    assert list(species["species_code"]) == ["gragoo", "soclap1"]
    assert len(respx_mock.calls) == 0


def test_load_frozen_recomputes_notable_flags(tmp_path):
    path = frozen_csv(tmp_path)
    assert list(species_list.load_frozen(notable("Vanellus gregarius"), path)["is_notable"]) == [False, True]
    assert list(species_list.load_frozen(notable("Anser anser"), path)["is_notable"]) == [True, False]


def test_load_frozen_rejects_notable_species_not_on_the_list(tmp_path):
    with pytest.raises(ValueError, match="Falco cherrug"):
        species_list.load_frozen(notable("Falco cherrug"), frozen_csv(tmp_path))


def test_load_frozen_without_a_list_explains_how_to_create_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="--refresh-species-list"):
        species_list.load_frozen(notable(), tmp_path / "missing.csv")


def test_diff_reports_added_removed_and_renamed():
    old = pd.DataFrame({"species_code": ["a1", "b1", "c1"], "common_name": ["A", "B", "C"],
                        "scientific_name": ["Aa a", "Bb b", "Cc c"], "family": ["F1", "F1", "F2"]})
    new = pd.DataFrame({"species_code": ["a1", "c1", "d1"], "common_name": ["A2", "C", "D"],
                        "scientific_name": ["Aa a", "Cc c", "Dd d"], "family": ["F1", "F3", "F2"]})
    changes = species_list.diff(old, new)
    assert changes == {"added": [("d1", "D")], "removed": [("b1", "B")],
                       "changed": [("a1", "common_name", "A", "A2"), ("c1", "family", "F2", "F3")]}
    assert species_list.describe_changes(changes) == [
        "  added:   d1 D", "  removed: b1 B", "  changed: a1 common_name: 'A' -> 'A2'",
        "  changed: c1 family: 'F2' -> 'F3'"]


def test_diff_of_identical_lists_is_empty():
    frame = pd.DataFrame({"species_code": ["a1"], "common_name": ["A"],
                          "scientific_name": ["Aa a"], "family": ["F1"]})
    assert species_list.diff(frame, frame) == {"added": [], "removed": [], "changed": []}


def test_refresh_writes_the_list_and_reports_changes(ebird_kz, tmp_path):
    path = frozen_csv(tmp_path)  # frozen: Graylag Goose + Sociable Lapwing only
    new, changes = species_list.refresh("KZ", notable("Vanellus gregarius"), path=path)
    assert len(new) == 5
    assert sorted(code for code, _ in changes["added"]) == ["demcra1", "eurnig1", "palsan1"]
    assert changes["removed"] == [] and changes["changed"] == []
    assert list(pd.read_csv(path)["species_code"]) == list(new["species_code"])  # written


def test_refresh_without_a_previous_list_has_no_diff(ebird_kz, tmp_path):
    new, changes = species_list.refresh("KZ", notable("Vanellus gregarius"), path=tmp_path / "new.csv")
    assert changes is None and (tmp_path / "new.csv").exists() and len(new) == 5


def test_the_committed_list_is_the_frozen_source_of_truth():
    """The shipped notable species all load against the committed list."""
    species = species_list.load_frozen(species_list.load_notable())
    assert species["species_code"].is_unique and species["is_notable"].sum() == 8
