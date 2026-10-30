"""Build and open the Chroma vector store (data/index/).

Each chunk's embedding is computed here, not by Chroma, so the model and its
settings are in our hands. The collection records which model and chunk settings
built it, and opening it with a different model configured raises an error:
query embeddings from another model would silently return nonsense. It also
records a fingerprint of the corpus file it was built from, and opening it when the
corpus has changed raises an error, so a stale index (e.g. one copied into a Docker
image) fails at startup instead of answering from old text.
"""

import hashlib
import shutil
from pathlib import Path

import chromadb
from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.config import Settings

from chattfinch import chunking, config, embedding

BATCH_SIZE = 1000


def _client(path: Path) -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(path), settings=Settings(anonymized_telemetry=False))


def chunk_metadata(chunk: chunking.Chunk) -> dict:
    """Everything about a chunk except its text: Chroma stores this next to the
    vector, and retrieval returns it with every hit."""
    return {"species_code": chunk.species_code, "common_name": chunk.common_name,
            "scientific_name": chunk.scientific_name, "source": chunk.source,
            "section": chunk.section, "url": chunk.url, "chunk_index": chunk.chunk_index,
            "n_tokens": chunk.n_tokens}


def corpus_fingerprint(corpus_path: Path = config.CORPUS_PATH) -> str:
    """SHA-256 of the corpus file. The corpus is gzipped deterministically, so an
    unchanged corpus keeps its fingerprint across rebuilds."""
    return hashlib.sha256(corpus_path.read_bytes()).hexdigest()


def build_index(chunks: list[chunking.Chunk], path: Path = config.INDEX_DIR,
                show_progress: bool = False, corpus_sha256: str | None = None) -> chromadb.Collection:
    """Embed `chunks` and store them in a fresh index at `path`. A previous Chroma
    store there is deleted first: deleting just the collection leaves its vector
    files behind, so each rebuild would grow the directory. `corpus_sha256` is the
    fingerprint of the corpus the chunks came from (see open_collection)."""
    if (path / "chroma.sqlite3").exists():  # only ever delete what is a Chroma store
        # Chroma caches one client per path in this process; drop it, or it would
        # still believe the deleted collection exists.
        SharedSystemClient.clear_system_cache()
        shutil.rmtree(path)
    client = _client(path)
    metadata = {"embedding_model": config.EMBEDDING_MODEL, "chunk_tokens": config.CHUNK_TOKENS,
                "chunk_overlap_tokens": config.CHUNK_OVERLAP_TOKENS}
    if corpus_sha256:
        metadata["corpus_sha256"] = corpus_sha256
    collection = client.create_collection(
        config.INDEX_COLLECTION, configuration={"hnsw": {"space": "cosine"}}, metadata=metadata)
    vectors = embedding.embed([c.text for c in chunks], show_progress=show_progress)
    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start:start + BATCH_SIZE]
        collection.add(ids=[c.chunk_id for c in batch],
                       embeddings=vectors[start:start + BATCH_SIZE].tolist(),
                       documents=[c.text for c in batch],
                       metadatas=[chunk_metadata(c) for c in batch])
    return collection


def open_collection(path: Path = config.INDEX_DIR,
                    corpus_path: Path | None = config.CORPUS_PATH) -> chromadb.Collection:
    """The built collection. Raises FileNotFoundError if there isn't one (run
    scripts/build_index.py), and ValueError if it was built with another model or
    from a corpus other than the one at `corpus_path` (None skips that check, for
    throwaway test indexes)."""
    client = _client(path)
    existing = {getattr(c, "name", c) for c in client.list_collections()}
    if config.INDEX_COLLECTION not in existing:
        raise FileNotFoundError(f"No index at {path}; run scripts/build_index.py")
    collection = client.get_collection(config.INDEX_COLLECTION)
    built_with = (collection.metadata or {}).get("embedding_model")
    if built_with != config.EMBEDDING_MODEL:
        raise ValueError(f"Index at {path} was built with {built_with!r} but config says "
                         f"{config.EMBEDDING_MODEL!r}; rebuild with scripts/build_index.py")
    if corpus_path is not None:
        built_from = (collection.metadata or {}).get("corpus_sha256")
        if built_from != corpus_fingerprint(corpus_path):
            raise ValueError(f"Index at {path} is stale: it was not built from the current "
                             f"{corpus_path.name}; rebuild with scripts/build_index.py")
    return collection
