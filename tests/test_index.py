"""Index and retrieval tests. They use the real embedding model (a few seconds to
load); a small throwaway index covers behavior, and the built index in data/index/
(skipped if absent) covers the plan's day-2 checks."""

import pandas as pd
import pytest

from chattfinch import chunking, config, corpus, embedding, index, retrieval


def make_doc(code, common, scientific, **sections):
    return {"species_code": code, "common_name": common, "scientific_name": scientific,
            "sections": [{"source": "wikipedia", "section": name, "text": text,
                          "url": f"https://example.org/{code}"} for name, text in sections.items()]}


DOCS = [
    make_doc("sakfal1", "Saker Falcon", "Falco cherrug",
             Feeding="The saker falcon hunts ground squirrels, pikas and pigeons on open steppe.",
             Status="The saker falcon is endangered because of trapping for falconry."),
    make_doc("demcra1", "Demoiselle Crane", "Grus virgo",
             Migration="Demoiselle cranes migrate in flocks over the Himalayas to winter in India."),
    make_doc("eurnig1", "Eurasian Nightjar", "Caprimulgus europaeus",
             Feeding="The nightjar catches moths and beetles in flight at dusk."),
]


@pytest.fixture(scope="module")
def small_index(tmp_path_factory):
    path = tmp_path_factory.mktemp("index")
    chunks = chunking.chunk_corpus(DOCS, embedding.count_tokens, 200, 30)
    return chunks, index.build_index(chunks, path), path


def test_every_chunk_is_stored_with_its_metadata(small_index):
    chunks, collection, _ = small_index
    assert collection.count() == len(chunks) == 4
    stored = collection.get(ids=[c.chunk_id for c in chunks], include=["documents", "metadatas"])
    by_id = dict(zip(stored["ids"], zip(stored["documents"], stored["metadatas"])))
    for chunk in chunks:
        text, metadata = by_id[chunk.chunk_id]
        assert text == chunk.text and text.strip()
        assert metadata == index.chunk_metadata(chunk)
        assert metadata["species_code"] == chunk.species_code and metadata["source"] == "wikipedia"


def test_retrieve_finds_the_matching_species(small_index):
    _, collection, _ = small_index
    hits = retrieval.retrieve("what prey does the saker falcon eat", k=3, collection=collection)
    assert hits[0]["metadata"]["species_code"] == "sakfal1"
    assert hits[0]["metadata"]["section"] == "Feeding"
    hits = retrieval.retrieve("where do demoiselle cranes migrate", k=1, collection=collection)
    assert hits[0]["metadata"]["species_code"] == "demcra1"


def test_hit_shape_and_ordering(small_index):
    _, collection, _ = small_index
    hits = retrieval.retrieve("moths at dusk", k=4, collection=collection)
    assert len(hits) == 4
    assert [h["score"] for h in hits] == sorted((h["score"] for h in hits), reverse=True)
    assert all(-1.0 <= h["score"] <= 1.0 for h in hits)
    assert set(hits[0]) == {"id", "text", "score", "metadata"}
    assert hits[0]["text"].startswith("Eurasian Nightjar (Caprimulgus europaeus) — Feeding\n")


def test_k_larger_than_the_index_returns_everything(small_index):
    _, collection, _ = small_index
    assert len(retrieval.retrieve("birds", k=50, collection=collection)) == 4


def test_species_filter_restricts_results(small_index):
    _, collection, _ = small_index
    hits = retrieval.retrieve("what do they eat", k=4, species_filter="eurnig1", collection=collection)
    assert {h["metadata"]["species_code"] for h in hits} == {"eurnig1"}
    hits = retrieval.retrieve("what do they eat", k=4, species_filter=["sakfal1", "demcra1"],
                              collection=collection)
    assert {h["metadata"]["species_code"] for h in hits} == {"sakfal1", "demcra1"}
    assert len(hits) == 3  # only 3 chunks belong to those two species


def test_species_filter_overrides_a_better_match_elsewhere(small_index):
    _, collection, _ = small_index
    hits = retrieval.retrieve("saker falcon hunting ground squirrels", k=1,
                              species_filter="demcra1", collection=collection)
    assert hits[0]["metadata"]["species_code"] == "demcra1"


def test_empty_species_filter_or_unknown_species_returns_nothing(small_index):
    _, collection, _ = small_index
    assert retrieval.retrieve("birds", species_filter=[], collection=collection) == []
    assert retrieval.retrieve("birds", species_filter="nosuch1", collection=collection) == []


@pytest.mark.parametrize("query, k", [("", 5), ("   ", 5), ("birds", 0), ("birds", -1)])
def test_invalid_arguments_raise(small_index, query, k):
    _, collection, _ = small_index
    with pytest.raises(ValueError):
        retrieval.retrieve(query, k=k, collection=collection)


def test_rebuilding_replaces_the_collection(small_index):
    chunks, _, path = small_index
    rebuilt = index.build_index(chunks[:2], path)
    assert rebuilt.count() == 2
    assert index.open_collection(path, corpus_path=None).count() == 2
    assert len([d for d in path.iterdir() if d.is_dir()]) == 1  # no orphaned vector files


def test_rebuild_refuses_to_delete_a_directory_that_is_not_a_chroma_store(small_index, tmp_path):
    chunks, _, _ = small_index
    (tmp_path / "precious.txt").write_text("keep me")
    index.build_index(chunks[:1], tmp_path)
    assert (tmp_path / "precious.txt").read_text() == "keep me"


def test_opening_a_missing_index_explains_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index.py"):
        index.open_collection(tmp_path / "nothing")


def test_opening_an_index_built_with_another_model_raises(tmp_path, monkeypatch):
    client = index._client(tmp_path)  # same settings as open_collection: Chroma refuses a second kind
    client.create_collection(config.INDEX_COLLECTION, metadata={"embedding_model": "other/model"})
    with pytest.raises(ValueError, match="other/model"):
        index.open_collection(tmp_path)


def test_opening_an_index_built_from_another_corpus_raises(small_index, tmp_path):
    chunks, _, _ = small_index
    corpus_file = tmp_path / "corpus.jsonl.gz"
    corpus_file.write_bytes(b"version 1")
    path = tmp_path / "index"
    index.build_index(chunks[:1], path, corpus_sha256=index.corpus_fingerprint(corpus_file))
    assert index.open_collection(path, corpus_path=corpus_file).count() == 1
    corpus_file.write_bytes(b"version 2")
    with pytest.raises(ValueError, match="stale"):
        index.open_collection(path, corpus_path=corpus_file)


def test_an_index_without_a_corpus_fingerprint_counts_as_stale(small_index, tmp_path):
    chunks, _, _ = small_index
    corpus_file = tmp_path / "corpus.jsonl.gz"
    corpus_file.write_bytes(b"anything")
    path = tmp_path / "index"
    index.build_index(chunks[:1], path)  # built before fingerprints were recorded
    with pytest.raises(ValueError, match="stale"):
        index.open_collection(path, corpus_path=corpus_file)


# --- The built index in data/index/ (the plan's day-2 checks) -------------------------

needs_built_index = pytest.mark.skipif(
    not (config.INDEX_DIR / "chroma.sqlite3").exists() or not (config.DUMPS_DIR / "corpus.jsonl.gz").exists(),
    reason="index not built; run scripts/build_index.py")


@pytest.fixture(scope="module")
def built():
    documents = corpus.read_corpus(config.DUMPS_DIR / "corpus.jsonl.gz")
    chunks = chunking.chunk_corpus(documents, embedding.count_tokens, config.CHUNK_TOKENS,
                                   config.CHUNK_OVERLAP_TOKENS)
    return chunks, index.open_collection()


@needs_built_index
def test_built_index_matches_the_current_corpus_and_chunker(built):
    """Fails when the corpus, chunker or chunk settings changed since the index was
    built: rebuild with scripts/build_index.py."""
    chunks, collection = built
    assert collection.count() == len(chunks)  # embedding count == chunk count
    assert set(collection.get(include=[])["ids"]) == {c.chunk_id for c in chunks}


@needs_built_index
def test_no_empty_chunks_and_none_over_the_token_limit(built):
    chunks, collection = built
    tokenizer = embedding.get_model().tokenizer
    for chunk in chunks:
        body = chunk.text.split("\n", 1)[1]
        assert body.strip(), chunk.chunk_id
        real = len(tokenizer(chunk.text)["input_ids"])  # with special tokens, as embedded
        assert real == chunk.n_tokens <= config.CHUNK_TOKENS <= embedding.get_model().max_seq_length


@needs_built_index
def test_every_species_has_chunks_and_metadata_survives(built):
    chunks, collection = built
    stored = collection.get(include=["documents", "metadatas"])
    assert all(doc.strip() for doc in stored["documents"])
    required = {"species_code", "common_name", "scientific_name", "source", "section", "url",
                "chunk_index", "n_tokens"}
    assert all(set(m) == required and m["url"].startswith("https://") for m in stored["metadatas"])
    species_with_chunks = {m["species_code"] for m in stored["metadatas"]}
    species_list = pd.read_csv(config.DUMPS_DIR / "species_list.csv")
    assert species_with_chunks == set(species_list["species_code"]) == {c.species_code for c in chunks}
    assert {m["source"] for m in stored["metadatas"]} == {"ebird", "wikipedia"}


@needs_built_index
@pytest.mark.parametrize("query, expected_species, top", [
    ("diet of the saker falcon", "sakfal1", 5),
    ("where do demoiselle cranes migrate", "demcra1", 1),
    ("how threatened is the sociable lapwing", "soclap1", 1),
    ("where does the great bustard live", "grebus1", 3),
])
def test_sanity_queries_find_the_right_species(built, query, expected_species, top):
    hits = retrieval.retrieve(query, k=top)
    assert expected_species in {h["metadata"]["species_code"] for h in hits}


@needs_built_index
def test_species_filter_on_the_built_index(built):
    hits = retrieval.retrieve("what does it eat", k=5, species_filter="eurnig1")
    assert len(hits) == 5
    assert {h["metadata"]["species_code"] for h in hits} == {"eurnig1"}


@needs_built_index
def test_known_limitation_alias_in_query_is_not_resolved_by_embeddings_alone(built):
    """Documents why day 4 added name resolution: 'goatsucker' is in the nightjar's
    profile chunk, yet plain semantic search doesn't surface the nightjar. The next
    test shows resolution fixing it."""
    hits = retrieval.retrieve("what does the common goatsucker eat", k=5)
    assert "eurnig1" not in {h["metadata"]["species_code"] for h in hits}


@needs_built_index
def test_name_resolution_finds_the_bird_semantic_search_misses(built):
    from chattfinch import rag, resolve
    query = "what does the common goatsucker eat"
    hits = rag.search(query, 5, resolve.resolve(query).species)
    assert {h["metadata"]["species_code"] for h in hits} == {"eurnig1"}
