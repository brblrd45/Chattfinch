# The Telegram bot. Build and run (from the repo root):
#
#   .venv/bin/python scripts/build_index.py      # if the corpus changed; the build checks
#   docker build -t chattfinch .
#   docker run -d --name chattfinch --restart unless-stopped --env-file .env \
#     -v "$PWD/data/private:/app/data/private:ro" chattfinch
#
# The mount gives the bot the IUCN snapshot for conservation-status answers. The
# snapshot is deliberately not in the image (IUCN terms forbid redistributing it);
# without the mount, status questions get a "not available" reply and everything
# else works.
# Never run two containers (or a container and a local bot) with the same bot token:
# only one process may long-poll a token, and a second makes both fail.
#
# API keys are NOT set here. Pass them at runtime with --env-file, as above.
# Anything in ENV, ARG, or a COPY'd file is readable by anyone with the image.
# No EXPOSE: long polling makes only outbound connections.

# Matches the local .venv (3.13); pyproject requires >=3.11.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Keys come from the environment (--env-file), never from a .env file in the image.
    PYTHON_DOTENV_DISABLED=1 \
    HF_HOME=/opt/huggingface

WORKDIR /app

# Dependencies first so code changes don't invalidate this layer. CPU-only torch
# first: otherwise sentence-transformers pulls the default Linux torch build with
# its CUDA libraries, several GB that a CPU server never uses.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Download the embedding model at build time, so the container starts without
# network access to Hugging Face, then stay offline at runtime. Must match
# EMBEDDING_MODEL in src/chattfinch/config.py: if it doesn't, the bot fails at startup
# (the model isn't in the cache and downloads are off) rather than run another model.
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL}')"
ENV HF_HUB_OFFLINE=1

# config.py finds data/ relative to its own path (parents[2]), so run from
# source via PYTHONPATH rather than pip-installing into site-packages.
COPY src/ src/
COPY scripts/ scripts/
COPY data/ data/
ENV PYTHONPATH=/app/src
# The vector index (data/index/, ~67 MB) is included on purpose so the bot starts
# without re-embedding the corpus. It is git-ignored, so it only exists if
# `scripts/build_index.py` has been run on the build machine. Fail here, with a
# clear message, if it is missing, built with another model, or stale (built from
# a different corpus than the one being copied in), rather than ship an image that
# can't answer or answers from old text.
RUN test -f data/index/chroma.sqlite3 \
    || (echo "data/index/ is missing: run scripts/build_index.py before docker build" >&2; exit 1)
RUN python -c "from chattfinch import index; index.open_collection()"

# Don't run as root. Chroma writes to its SQLite file, so the app owns /app.
RUN useradd --create-home --uid 1000 chattfinch \
    && chown -R chattfinch:chattfinch /app
USER chattfinch

CMD ["python", "-m", "chattfinch.bot"]
