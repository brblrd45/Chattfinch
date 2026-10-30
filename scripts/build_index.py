"""Chunk the corpus, embed the chunks, and build the vector store in data/index/.

Run from the repo root:  .venv/bin/python scripts/build_index.py

Reads data/dumps/corpus.jsonl.gz (built by build_dumps.py). Takes a minute or two;
downloads the embedding model on the first run.
"""

import statistics
import time

from chattfinch import chunking, config, corpus, embedding, index


def dir_size_mb(path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


def main() -> None:
    documents = corpus.read_corpus(config.CORPUS_PATH)
    chunks = chunking.chunk_corpus(documents, embedding.count_tokens, config.CHUNK_TOKENS,
                                   config.CHUNK_OVERLAP_TOKENS)
    sizes = [c.n_tokens for c in chunks]
    print(f"{len(documents)} documents -> {len(chunks)} chunks "
          f"(tokens: median {statistics.median(sizes):.0f}, max {max(sizes)}; "
          f"model: {config.EMBEDDING_MODEL}, device {embedding.get_model().device})")
    started = time.time()
    collection = index.build_index(chunks, show_progress=True,
                                   corpus_sha256=index.corpus_fingerprint(config.CORPUS_PATH))
    print(f"Embedded and stored {collection.count()} chunks in {time.time() - started:.0f}s; "
          f"{config.INDEX_DIR.relative_to(config.ROOT)} is {dir_size_mb(config.INDEX_DIR):.1f} MB")


if __name__ == "__main__":
    main()
