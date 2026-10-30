"""Semantic retrieval over the vector store."""

from functools import lru_cache

import chromadb

from chattfinch import embedding, index


@lru_cache(maxsize=1)
def _default_collection() -> chromadb.Collection:
    return index.open_collection()


def warm_up() -> None:
    """Load the embedding model and open the index now, instead of on the first
    query. Raises what index.open_collection raises if the index is missing or was
    built with another model."""
    embedding.get_model()
    _default_collection()


def retrieve(query: str, k: int = 5, species_filter: str | list[str] | None = None,
             collection: chromadb.Collection | None = None) -> list[dict]:
    """The `k` chunks most similar to `query`, best first.

    `species_filter` is a species code or list of codes: only those species'
    chunks are searched (day 4 uses this once a name in the query is resolved).
    Each hit is {"id", "text", "score", "metadata"}; `score` is cosine similarity
    (1.0 = identical direction), `metadata` has species_code, common_name,
    scientific_name, source, section, url, chunk_index and n_tokens.
    """
    if not query.strip():
        raise ValueError("Empty query")
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    where = None
    if species_filter is not None:
        codes = [species_filter] if isinstance(species_filter, str) else list(species_filter)
        if not codes:
            return []
        where = {"species_code": {"$in": codes}}
    collection = collection or _default_collection()
    result = collection.query(query_embeddings=embedding.embed([query]).tolist(), n_results=k,
                              where=where, include=["documents", "metadatas", "distances"])
    return [{"id": id_, "text": text, "score": 1.0 - distance, "metadata": metadata}
            for id_, text, metadata, distance in zip(result["ids"][0], result["documents"][0],
                                                     result["metadatas"][0], result["distances"][0])]
