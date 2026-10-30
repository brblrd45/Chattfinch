# "Ask a Birder" — One-Week Build Plan

A RAG assistant over the birds of **Kazakhstan**, with citations, name-resolved retrieval, query routing, a reranker, and a real evaluation harness. Handles the "same bird, multiple English names" problem so a query for "common goatsucker" finds the same species as "common nightjar." Scoped so a working demo exists by day 3 and the differentiators land by day 7.

**Interface: a Telegram bot.** A user messages the bot a question and it replies with the answer and its sources. (The plan originally called for a Streamlit web app; that was built on day 3 and is kept only as a local inspection tool. See Day 3.)

## Scope: birds of Kazakhstan

- **Region:** Kazakhstan (eBird region code `KZ`). eBird lists 502 taxa for `KZ`: **491 species** plus 11 hybrids (eBird's list changes over time: it was 490 species when first pulled, so the list is frozen; see below). By default the target is the **full eBird `KZ` species list** (category "species" only), plus a hand-picked set of regionally notable species (e.g. Sociable Lapwing, Saker Falcon, Demoiselle Crane, Great Bustard, Dalmatian Pelican, Steppe Eagle, Black Lark, Pallas's Sandgrouse). The list is **frozen**: the committed `species_list.csv` is the source of truth and the build never calls eBird for it. To update it deliberately, run `build_dumps.py --refresh-species-list`: it re-pulls the list, prints what was added, removed or renamed, and rewrites the file, for review with `git diff` before committing.
- **Frequency ranking is optional and off by default.** eBird coverage of Kazakhstan is too sparse for reporting frequencies to be meaningful, so the list isn't ranked or cut. The ranking code still exists for regions with good eBird coverage: set `FREQUENCY_FILE` in `config.py` to a bar-chart export downloaded from the eBird website, and the list is cut to the top `TARGET_SPECIES_COUNT` (default 200) plus the notable species. **Not usable yet:** the bar-chart parser isn't written.
- **Optional (not implemented yet): filter by an Avibase checklist.** If the full list makes manual review (alias QC, coverage misses) too heavy, download a Kazakhstan checklist from Avibase (a one-click manual download, no API), save it under `data/manual/`, and keep only eBird species that also appear on it, matched by scientific name. This would be a second optional config setting alongside `FREQUENCY_FILE`, off by default.
- **Why Kazakhstan works well here:** it sits on the Central Asian Flyway, with steppe, desert, wetland and mountain birds, so range, migration and conservation-status questions are rich. It also has many Eurasian species whose English names vary between sources ("Common"/"Eurasian"/"European" prefixes, US vs. UK usage), which makes the alias problem real rather than contrived.
- **Language:** English names only for the app. Scientific names are the anchor for alias resolution. Russian and Kazakh vernacular names are out of scope; list them under "what I'd do with more time."
- **Out-of-scope behavior:** birds not on the list (including any Kazakhstan birds excluded from it) should get a clear "not in my coverage" answer, not a hallucination or an answer from general knowledge.

**Data sources (all pull cleanly):**
- eBird API (taxonomy, occurrence/range, and the `KZ` regional species list) — free key
- GBIF API (open, no key — taxonomy and synonym resolution via name matching, plus vernacular names; the noisiest name source)
- IUCN Red List API v4 (conservation status, plus English common names) — free token, **requires an approval request** (see Day 0)
- Wikipedia, via the MediaWiki Action API directly (descriptive corpus; article titles and redirects are a source of alternate English names, but a noisy one). Wikimedia throttles requests whose user agent has no real contact address, so `WIKIPEDIA_USER_AGENT` in `.env` must have one.

**Canonical species ID.** Use the **eBird/Clements species code** (e.g. `eurnig1` for Eurasian Nightjar) as the canonical ID, with the scientific name stored alongside. eBird, GBIF and IUCN disagree on some genus placements, splits and lumps; when a source's taxon doesn't map 1:1 to a Clements species, record it in a `taxonomy_mismatches.csv` log and resolve it by hand rather than silently dropping it. Known so far: GBIF can't match 8 of eBird's newer genus names (*Astur*, *Tachyspiza*, *Anarhynchus*, ...), and IUCN has no record for 23 species even after retrying with GBIF's accepted name (e.g. Demoiselle Crane is *Grus virgo* in eBird but *Anthropoides virgo* in GBIF and IUCN; the retry fixes that one).

**Data strategy: dumps vs. live API.**
The rule is: **ship what's compact and near-invariant as small dated dumps; call live only what's genuinely dynamic.** That split is also the strong architecture: an offline batch job that pulls from APIs, normalizes, and writes small dumps, and an online app that reads those dumps at query time. Call this out explicitly in the README; the ingestion-vs-serving separation is exactly what applied-LLM roles look for.

Dumps live **in the local git repo**, not behind a separate server, so the bot has no dependency on any API being up *or* on a second host. **Only the source code is shared publicly; the data files are not** (dumps, corpus, index, IUCN snapshot). Anyone running the shared code regenerates the data with `build_dumps.py` and `build_index.py` and their own API keys (see Day 7).

| Data | Where | Why | Rough size |
|---|---|---|---|
| Name-alias index: `alias_index.json` (`{alias → [canonical_id, ...]}`) + `aliases.csv` (each alias with its species, kind and sources) | **Dump (repo)** | Small, invariant during the project | ~100 KB |
| Taxonomy table: `species_list.csv` (491 species) | **Dump (repo)** | Small, stable | ~50 KB |
| Raw API responses (`data/raw/`) | **Local cache, not committed** | Regenerable; lets reruns skip finished fetches | MB |
| IUCN status snapshot: `data/private/iucn_snapshot.json` (with fetch dates) | **Local only, never committed** | Changes on the order of years; a dated snapshot is honest and keeps a keyed API off the hot path. **IUCN's terms forbid redistribution**, so it lives in the gitignored `data/private/` (the pre-commit hook also blocks it) and a fresh clone regenerates it with its own IUCN token via `build_dumps.py`. Test fixtures for IUCN are synthetic for the same reason | ~100 KB |
| Corpus: `corpus.jsonl.gz` (per-species, source-tagged sections) | **Dump (repo)** | Frozen once ingested; modest as compressed text. Wikipedia text is CC BY-SA 4.0: every document records its article's revision permalink, retrieval date and license, for attribution. No IUCN data in it | low MB |
| Embeddings / vector store | **Commit only if small; else regenerate** | The one thing that can get big. If it's too large, ship the corpus + a one-command `build_index.py` and rebuild locally | MB–tens of MB |
| eBird recent-sightings / occurrence | **Live API** | Genuinely dynamic ("seen lately"), and large — exactly what not to store | — |

Keep at least one live API in the query path (eBird recent observations for `KZ` is the natural one) so the project still demonstrates live third-party integration. It must fail gracefully: if eBird is down or rate-limited, the app says so and still answers everything else. Every dump has a documented regeneration script rather than being a mystery blob.

**Baseline environment (install day 1):**
```
python 3.11+
httpx                     # API calls (one HTTP stack for every client)
respx                     # mocks httpx in tests
pandas                    # data wrangling
python-dotenv             # API keys
sentence-transformers     # embeddings + the day-6 cross-encoder reranker
chromadb                  # vector store
rank-bm25  (optional)     # keyword retrieval for hybrid search; not installed (day 4, step 5)
anthropic                 # generation (claude-haiku-5-5; see Day 3)
python-telegram-bot       # the user interface: a Telegram bot
streamlit  (optional)     # local inspection view only, not deployed
pytest                    # tests
ragas  (optional, day 5)  # LLM-as-judge eval; pulls datasets + a judge model
```

**Embedding model and chunk size must match.** `all-MiniLM-L6-v2` truncates input at 256 word pieces, so chunks longer than that are silently embedded from their first part only. Either chunk at **≤200–256 tokens** with that model, or use a model with a 512-token limit (e.g. `bge-small-en-v1.5`) and chunk at up to ~400. **Chosen on day 2:** `all-MiniLM-L6-v2` with 200-token chunks; write the choice and its limit in the README.

---

## Day 0 — Pre-work (before the week starts, ~1 hour) ✅ done

- ✅ Request the **IUCN Red List API token now**; approval can take days. Fallback if it hasn't arrived by day 1: use the conservation-status fields available via GBIF/Wikidata for the snapshot, and note the substitution in the README.
- ✅ Register for the eBird key. Create the repo, `.env.example`, a `.gitignore` covering every `.env` variant, and the pre-commit hook in `scripts/hooks/` that blocks staged `.env` files and any `.env` value in a diff (enable with `git config core.hooksPath scripts/hooks`).
- ✅ IUCN terms read: **no redistribution**, so IUCN data stays out of the repo (see the dump table).
- ✅ Wikipedia reuse terms: closed. Only the source code is shared; the corpus and other data files are not published. The bot's answers still quote Wikipedia text to users, so every cited source keeps its attribution (article, revision link, CC BY-SA 4.0), as built on day 3.

---

## Day 1 — Data ingestion, alias index, corpus

**Status:** ✅ day 1 complete. Run everything with `.venv/bin/python scripts/build_dumps.py`.

**Implement**
- ✅ Thin API clients (`src/chattfinch/clients/`): eBird, GBIF, IUCN and Wikipedia, on a shared `httpx` client with retries, backoff (honoring `Retry-After`) and rate limiting. Behaviors found in the real APIs:
  - GBIF answers an unknown name with a `HIGHERRANK` match to Aves, not "no match", so only species-rank matches count.
  - A species' newest IUCN assessment can be regional (Sociable Lapwing: Europe 2021), so only the latest **global** assessment is used (CR, 2019).
  - IUCN returns 404 for names it files differently; retry with GBIF's accepted name.
  - A Wikipedia lookup by scientific name can land in a *section* of a genus article; that article's redirects then belong to other taxa and are ignored.
- ✅ Build `species_list.csv` (see Scope) from the eBird `KZ` species list, keeping taxonomy category "species" only, and flag the notable species (the build fails if a notable species isn't on the eBird list). Frequency ranking stays off unless `FREQUENCY_FILE` is set; its bar-chart parser isn't written yet.
  - **List frozen:** eBird's list drifts (White-winged Scoter, *Melanitta deglandi*, appeared between day 1 and day 2, taking 490 species to 491), so `species_list.csv` is committed and read as-is. `--refresh-species-list` re-pulls it, reports the changes and rewrites the file; after a refresh, rebuild the rest and the index (the staleness tests fail until then). `is_notable` is recomputed from `notable_species.csv` on every load, and a notable species missing from the frozen list is an error.
- ✅ **Build the name-alias index** (`aliases.py`, raw names fetched and cached per species by `name_sources.py`). Sources: eBird common name, scientific names (eBird's plus GBIF/IUCN names when they differ, e.g. older genus placements), GBIF vernacular names, IUCN English names, and the Wikipedia article title + redirects. Outputs `alias_index.json` and `aliases.csv`.
  - **Normalize** every alias the same way at build time and query time: lowercase, strip diacritics, drop possessive *'s*, hyphens to spaces, fold simple plurals.
  - **Clean raw names:** split comma lists ("Cuckoo, Common Cuckoo"), keep multi-word parentheticals ("Rock Pigeon (Feral Pigeon)"), drop slash names ("Greater/Lesser Scaup"), hybrids and banding codes ("EUJA").
  - **Group aliases:** a one-word name that is the head noun of 2+ species ("warbler", "nightjar", "dipper") maps to all of them, marked `kind = group`, instead of to whichever species happened to list it.
  - **Collision rules:** an eBird common or scientific name belongs only to its own species (a stale Wikipedia redirect sent "Buteo japonicus" to Common Buzzard); a GBIF-only pairing is dropped when another source confirms the same name for a different species ("grey plover": IUCN-backed for Black-bellied Plover, GBIF-only for Asian Houbara).
  - **Ambiguity policy:** genuinely ambiguous aliases keep all candidates; the bot asks which one the user means, with one button per candidate species (day 4), rather than guessing.
  - **Manual review:** read `aliases.csv`; fix anything wrong in `data/manual/alias_overrides.csv` (`add`/`remove` rows, applied last).
- ✅ **IUCN status snapshot** from the cached IUCN records, written to `data/private/iucn_snapshot.json` (never committed; see the dump table) with the date range the records were fetched. Latest global category per species: 417 LC, 24 NT, 17 VU, 7 EN, 3 CR, and 23 with no IUCN record under eBird's or GBIF's name.
- ✅ `taxonomy_mismatches.csv` (committed; names and issue types only): 66 rows where GBIF, IUCN or Wikipedia disagree with eBird's taxonomy, for hand review. The 23 species missing from IUCN are the ones to review first, since the app can't give their status.
- ✅ **Corpus** (`corpus.py`): one document per species with source-tagged sections. Each starts with an `ebird` *Profile* section generated from the taxonomy and the species' own aliases ("Also known as: ..."), so retrieval can find the species under any name; then the Wikipedia article's sections (reference/link sections dropped, subsections as "Behaviour > Feeding"), each citing the exact revision permalink. Each doc carries its canonical ID and aliases. IUCN status is not in the corpus (the corpus is committed, and IUCN data isn't redistributable); the app reads status from the private snapshot at query time (day 4 router). Output is gzipped deterministically, so an unchanged corpus rebuilds byte-identical.
- ✅ `scripts/build_dumps.py` produces the artifacts and a `manifest.json` (build date, settings, row counts). This script *is* the regeneration story. Raw API responses are cached in `data/raw/` (not committed): reruns only fetch what's missing, a failed source is retried next run, and a source is skipped for the rest of a run after 5 consecutive failures.

**Test** (tests use dummy keys and trimmed recordings of real responses in `tests/fixtures/`)
- ✅ Unit tests for each client (mocked responses; assert schema and the API quirks above).
- ✅ Unit tests for each alias rule and for the name-source cache.
- ✅ Checks on the committed dumps (`tests/test_dumps.py`): unique species codes; every species reachable by its eBird and scientific names; **known name pairs** resolve to the same canonical ID (Eurasian/European Nightjar/Goatsucker, Sociable Lapwing/Plover, Grus/Anthropoides virgo, Black-bellied/Grey Plover).
- ✅ **Ambiguity check (replaces a "no collisions" assert):** every alias mapping to 2+ species must be a group alias or on a reviewed list; a new ambiguous alias fails the test until it's reviewed.
- ✅ Corpus coverage (`tests/test_dumps.py`): one document per species; every species has Wikipedia text with a revision permalink and license; misses fail the test with the species codes.

**End of day:** a `species_list.csv`, an alias index, an IUCN snapshot, a source-attributed corpus, and a working `build_dumps.py`.

---

## Day 2 — Dumps, chunking, embedding, vector store ✅ done

**Status:** ✅ complete. Build the index with `.venv/bin/python scripts/build_index.py` (about a minute on an Apple GPU, longer on CPU).

**Implement**
- ✅ **Publish the dumps:** the dumps are committed in `data/dumps/` (2.1 MB) and the app reads them from local paths. `DUMPS_BUDGET_MB` in `config.py` is checked by `build_dumps.py` and a test. The server allotment is about 15 GB shared with everything else, so the budget is set to 15,000 MB and acts as a loose guard against a runaway dump, not a tight limit (the dumps are ~2 MB). The dumps never go to GitHub, so its 100 MB file limit doesn't apply. `DATA_LICENSES.md` records Wikipedia attribution/ShareAlike, the IUCN decision, and what is still unverified (eBird and GBIF terms, Wikipedia reuse guidelines).
- ✅ **Chunker** (`chunking.py`): chunks of at most **200 tokens** including a one-line header ("Saker Falcon (Falco cherrug) — Behaviour > Feeding"), well under the model's 256 limit. The header names the species in every chunk, so a feeding paragraph that never repeats the bird's name can still match "diet of the saker falcon". Chunks are packed from whole sentences; a short boundary sentence repeats in the next chunk; a sentence over budget (e.g. the profile's alias list) is split at semicolons, then words. Sentences end only before a capital letter, so "c. 1758" and "ca. 25 cm" aren't split. Tokens are counted with the model's own tokenizer. Every chunk keeps `species_code`, `common_name`, `scientific_name`, `source`, `section`, `url` as metadata.
- ✅ **Embedding model:** `all-MiniLM-L6-v2` (256-token limit, 384 dimensions); the choice and chunk settings are in `config.py` and recorded in the index, and opening an index built with a different model raises an error rather than silently returning nonsense.
- ✅ **Vector store** (`index.py`): Chroma, cosine similarity, persisted in `data/index/`. 491 documents become **8,930 chunks** (median 169 tokens). A rebuild starts from an empty directory.
- ✅ **`retrieve(query, k, species_filter=None)`** (`retrieval.py`): returns `{id, text, score, metadata}` best first, with `score` = cosine similarity. `species_filter` takes a species code or a list, ready for day 4.
- ✅ **Where the vectors live: regenerated, not committed.** The store is 67 MB of binary files that change on every rebuild, which would bloat the git history by that much each time. `data/index/` is git-ignored; a fresh clone runs `build_index.py`. **The Docker image does include it:** `COPY data/` copies the locally built index in, so the container starts in seconds instead of re-embedding the corpus (rebuilds will be rare, so the extra 67 MB and the rebuild-on-change cost don't matter). The build fails with a clear message if the index hasn't been built, was built with another model, or is stale: the index records a SHA-256 of the corpus it was built from, and opening it against a different corpus raises an error (at `docker build` and again at bot startup). Before `docker build`, run `build_index.py` if the corpus changed. Changes to the chunker code alone aren't fingerprinted; `tests/test_index.py` catches those.

**Test**
- ✅ Chunker unit tests (16): budgets, header in every chunk, sentence integrity, overlap rules, oversized sentences, empty sections.
- ✅ Index and retrieval tests on a throwaway index with the real model, and checks on the built index: embedding count == chunk count (and the same IDs, so a stale index fails the test); no empty chunks; **no chunk over the token limit, verified with the real tokenizer**; all 491 species present; metadata survives the round trip.
- ✅ Hand sanity queries (also as tests):
  - "where do demoiselle cranes migrate" and "how threatened is the sociable lapwing": the right species ranks first (scores 0.71–0.75).
  - "diet of the saker falcon": the Saker's chunks fill places 2–5 but a Red-footed Falcon diet chunk ranks first.
  - "what does the common goatsucker eat": **nothing about nightjars comes back**, although the nightjar's profile chunk lists the name. Embeddings alone don't resolve aliases; that is what day 4's name resolution and species filter are for (a test documents this as a known limitation).
  - "tell me about penguins": best score 0.44. Off-topic scores are lower but not cleanly separated from poor on-topic ones (the goatsucker miss scores 0.51), so day 3's "decline" behavior can't rest on a score cutoff alone.

**End of day:** working semantic retrieval over the corpus, with committed dumps and a regenerable index.

---

## Day 3 — End-to-end RAG + Telegram bot  *(ship a working thing today)*

**Implement**
- ✅ **Prompt template** (`prompts.py`): system instructions + retrieved context + question. The model answers *only* from numbered `<source>` blocks and cites them as `[1]`; each block states the species' canonical name (plus scientific name, section and origin), so day 4's name surfacing is an incremental change. When the sources don't answer the question the reply starts with a fixed `NO_ANSWER` marker plus a reason (`bird_not_found`, `not_about_birds` or `detail_missing`), so code can tell a refusal from an answer without guessing from wording, and can choose the wording itself. For the first two the user sees a fixed message ("I couldn't find that bird in my sources. I only cover birds recorded in Kazakhstan, so it may not be one of them.") and the "Show passages" button is hidden; for `detail_missing` the model says in a sentence what the sources lack. This came out of the first real Telegram session: asked about the tufted titmouse, the model declined correctly but explained that its sources "cover only sandgrouse species", describing the unrelated passages the search had returned. The fixed message is hedged on purpose: until day 4 resolves names against the species list, a bird the search missed ("common goatsucker") is indistinguishable from a bird that isn't covered. Source text and the question are escaped, and the prompt tells the model to treat source text as reference material, not instructions.
- ✅ **`answer(query)`** (`rag.py`): retrieve (top 6 chunks) → build prompt → call LLM → return `Answer(text, declined, sources)`. The sources are exactly what the model saw, numbered as the answer cites them. Retrieval and the model are passed in as functions, so tests use fakes. The model client (`llm.py`) is the only provider-specific module: Anthropic `claude-haiku-5-5` (the cheapest current Claude model, chosen to keep costs down), effort `medium`; failures become one `LLMError` with a clear message. `scripts/ask.py "question"` asks from the command line.
- ✅ **Citation handling** (`citations.py`): finds the `[n]` markers in an answer (also `[1][3]` and `[1, 3]`), maps each to the source the model was shown, and labels it ("Saker Falcon (Falco cherrug) — Ecology · Wikipedia (CC BY-SA 4.0)"). Cited sources are listed under the answer with links to the exact Wikipedia revision; markers in the text become links; sources retrieved but not cited are kept separate. A number that matches no source shown (an invented citation) is reported, never silently resolved.
- ✅ **Streamlit app, now a local inspection tool only** (`src/chattfinch/app.py`, run with `.venv/bin/streamlit run src/chattfinch/app.py`): built before the switch to Telegram. Question box → answer with linked citations → cited sources → expandable panel with every retrieved passage and its similarity score. It stays useful for reading an answer against its passages on a big screen, but it is not deployed and nothing depends on it; delete it (with `tests/test_app.py`, `.streamlit/` and the `streamlit` requirement) if it isn't earning its keep by day 7.
- ✅ **Telegram bot, built, tested offline and run against real Telegram** (`src/chattfinch/bot.py`, run with `.venv/bin/python -m chattfinch.bot`; formatting in `telegram_format.py`): the interface people actually use. It reuses everything above unchanged (`rag.answer`, `citations`); only the presentation is new. Built on `python-telegram-bot` 22.8; every call the bot makes was checked against that library's real signatures. As built:
  - **Setup:** create the bot with @BotFather, put its token in `.env` as `TELEGRAM_BOT_TOKEN` (a secret: the pre-commit hook already blocks any `.env` value from being committed), add it to `.env.example`. To find a user ID, message the bot from an account not yet on the allowlist: its "this bot is private" reply states the ID. Add it to `TELEGRAM_ALLOWED_USER_IDS` and restart.
  - **Long polling, not a webhook.** The bot asks Telegram for new messages over an outbound connection, so the server needs no open port, no domain and no HTTPS certificate. Only one running process may poll a given bot token; a second one makes both fail.
  - **Flow:** any text message is a question → show "typing…" → `rag.answer` → reply. Each message is answered on its own (no conversation memory; multi-turn stays under "what I'd do with more time"). `/start` and `/help` explain what the bot covers and offer example questions as buttons. Private chats only. Questions over 500 characters are refused. Messages sent while the bot was down are discarded on restart, so it doesn't answer (and pay for) a stale backlog.
  - **Don't block the bot:** `rag.answer` is synchronous (embedding + model call, a few seconds), and the bot library is async, so run it in a worker thread; otherwise one slow answer freezes the bot for everyone. Answers are produced one at a time, which also bounds how fast the API key can be spent.
  - **Formatting:** Telegram's HTML mode, with the answer's `[n]` markers as links to the sources and the cited sources listed underneath with their labels (species, section, "Wikipedia (CC BY-SA 4.0)"), which is the attribution the license requires. Link previews off. Escape everything that isn't our own markup. Messages are capped at 4,096 characters: an answer plus sources fits, but passage text does not, so it goes in a separate message. If Telegram rejects a message's markup, the bot resends it as plain text rather than leave the user without an answer.
  - **"Show passages"**, the equivalent of the web app's expandable panel: a button under the answer that sends every retrieved passage with its similarity score, cited ones first, then removes itself. It costs nothing (the answer is looked up in the cache); if the answer has been evicted, the bot says to ask again.
  - **Declined answers, invented citations and errors** each get a plain message: the fixed decline messages described under the prompt template above (by `NO_ANSWER` reason), a caution when a citation matches no source, and a short apology for model or network errors (details go to the log, not the chat).
  - **Access control:** anyone who finds a bot's username can message it, and every question spends the Anthropic key. Only user IDs in `TELEGRAM_ALLOWED_USER_IDS` reach the model; everyone else gets a one-line "this bot is private" reply. An empty list means nobody is allowed (closed by default). Second guard: at most 20 new (uncached) questions per user per hour.
  - **Answer cache:** keep a small in-memory cache keyed by question text, so a repeated question doesn't pay for the model twice (what `st.cache_data` did in the web app).
  - **Privacy and secrets:** questions pass through Telegram and are sent to Anthropic. Logs record user IDs, timings and outcomes, never message text; say so in the README. The HTTP library's request logging is turned down because Telegram request URLs contain the bot token.
- ✅ **Dockerfile finished** (build and run commands are at the top of the `Dockerfile`).
  - ✅ `CMD` runs the bot (`python -m chattfinch.bot`). No `EXPOSE`: long polling makes only outbound connections. Run it with `docker run -d --restart unless-stopped --env-file .env chattfinch` so it comes back after a crash or reboot. Never run two containers (or a container and a local bot) with the same bot token.
  - ✅ Embedding model: downloaded during the image build, and the container runs with `HF_HUB_OFFLINE=1`, so it starts without reaching Hugging Face. The model name is a build `ARG` that must match `EMBEDDING_MODEL` in `config.py`; if it doesn't, startup fails rather than running another model.
  - ✅ Torch is installed from the CPU-only wheel index, so an x86 server doesn't pull several GB of CUDA libraries. Image: 3.24 GB on disk (765 MB compressed), built on arm64.
  - ✅ The container ignores `.env` files (`PYTHON_DOTENV_DISABLED=1`); keys come only from `--env-file`.
  - ✅ Stale-index check: the index records a SHA-256 of the corpus, and `open_collection` refuses a mismatch. It runs at `docker build` (the build fails) and at bot startup. Two tests cover it.
  - ✅ `data/private/` (the IUCN snapshot) stays out of the image; the mount (`-v "$PWD/data/private:/app/data/private:ro"`) is documented in the Dockerfile for when day 4's status answers need it.
  - ✅ Built and smoke-tested with no network and no keys: the model loads offline as the non-root user, the index opens, "where do demoiselle cranes migrate" returns Demoiselle Crane first (0.71), and the bot exits with a clear message when the token is missing.
  - ⬜ Run it against real Telegram with `--env-file .env` (needs my keys; stop any local bot first). If the server is x86, build there or with `--platform linux/amd64`; that build hasn't been tried.
- ✅ **Eval set started** (`eval/questions.jsonl`, loaded and validated by `evaluation.py`): 19 questions, each with its type, the expected **species**, the expected section(s) where the answer lives, and whether it should be declined. 12 in-scope questions across identification, behavior, range and status; 3 that use an alternate name ("common goatsucker", "Sociable Plover", "Falco cherrug"), two of them twins of a standard-name question so the pair can be compared; 4 out-of-scope. It is the single source of the questions: the live tests read it too. A test checks every species code and section name against the corpus, so a typo can't silently lower the scores.
  - **Retrieval baseline** (`scripts/eval_retrieval.py`, free, no model calls; top 6 chunks, semantic search only): right species ranked first for **12 of 12** standard-name questions and **0 of 3** alternate-name questions (goatsucker and *Falco cherrug* miss entirely; Sociable Plover is found at rank 6). Overall hit@1 80%, hit@6 87%, MRR 0.81, right section in the top 6 87%. This is the number day 4 has to beat, and it says where: names.
  - Out-of-scope similarity scores overlap in-scope ones (Black-necked Crane 0.64, Himalayan Monal 0.60, penguins 0.44), confirming that declining can't rest on a score cutoff.

**Test**
- ✅ End-to-end questions across types (ID, behavior, range, status), run live on 2026-10-13 with `.venv/bin/pytest tests/test_answer_live.py --live` (model: `claude-haiku-5-5`): **17 of 19 passed.** All 12 standard-name questions were answered with valid citations to the right species. The two failures are the alternate-name questions the retrieval baseline predicted, q13 ("common goatsucker") and q15 ("Falco cherrug"): retrieval never surfaced the right bird, and the model said so instead of answering from the wrong birds' text. That is the correct behavior for the model and a retrieval failure for the system; day 4's name resolution is the fix. q14 ("Sociable Plover") passed because the right chunk was retrieved at rank 6.
- ✅ Faithfulness, the mechanical part: all 4 out-of-scope questions were declined in the same run (penguins; Black-necked Crane, a neighboring-region bird not on the `KZ` list, where retrieval returns other cranes' text that must not be used; Himalayan Monal; a non-bird question), and no answer cited a source it wasn't shown.
- ✅ Faithfulness, the human part (done by hand): read a handful of answers sentence by sentence against their sources (`scripts/ask.py --show-text`). The automated checks confirm *that* an answer cites the right species, not that each sentence is supported by what it cites.
- ✅ Automated tests that need no key or index: prompt construction, the answer pipeline with a fake retriever and model, and the model client against a fake API (request shape, every failure mode), citation parsing and rendering, and the Streamlit app run headlessly with a fake pipeline.
- ✅ Bot tests, no Telegram connection needed (`tests/test_telegram_format.py`, `tests/test_bot.py`): message formatting (links, escaping of model and source text, the 4,096-character limit, passages split across messages without breaking markup), the handlers with fake updates and a fake pipeline (question → reply, declined, model error, empty and non-text messages, buttons, cache, rate limit), and access control (a user who isn't on the allowlist is refused by every handler and never reaches the model; disabling the check makes 7 tests fail).
- ✅ First manual run against real Telegram (2026-10-13): a relevant and an irrelevant question behaved as expected, and it surfaced the misleading decline wording fixed above. ✅ Second manual run (2026-10-18, against the Docker container, with day 4's name resolution): "Show passages" worked; a bird that isn't covered (tufted titmouse) got the decline message; "common goatsucker" got an answer about the Eurasian Nightjar; "cat owl" offered two buttons and answered for the owl picked; "How swift is the Peregrine Falcon" was about the Peregrine. No errors in the logs, and no Telegram URLs (so no bot token). ⬜ Still to do: a message from an account that isn't on the allowlist (needs a second person; check the logs for `refused user=…`), and `.venv/bin/pytest tests/test_answer_live.py --live`, since the prompt has changed twice.
- ✅ **Logging for debugging** (added after that run, because the first answer was logged as `declined=True` and the logs couldn't say why): each answer's log line now has the decline reason (`bird_not_found`, `not_about_birds`, `detail_missing`, or `-`), counts of birds named and renamed, and whether a bird was chosen from buttons. "Show passages" taps are logged too. Still no message text, and tests check that. The image is rebuilt; restart the container to pick it up.

**End of day:** a working Telegram bot and a seed eval file (19 questions). This is my fallback deliverable if the week goes sideways. *Status: the pipeline, citations, eval set and bot are done and verified live. Left: the allowlist check with a second account.*

---

## Day 4 — Name resolution, query routing (hybrid retrieval optional)

**Order of work: name resolution → router → cache → BM25 (left optional; see step 5).** The day-3 baseline shows where the gap is: right species ranked first for 12 of 12 standard-name questions but 0 of 3 alternate-name ones. Name resolution fixes that gap directly, so it goes first. Building the stages in this order also gives each one its own row in the ablation, so it's clear what each one adds. This day is overloaded: **if it overruns, BM25 + fusion moves to the day-6 buffer**; name resolution and the alias eval questions don't.

**Implement**
1. **Alias eval questions first** (retrieval-only, so free to run). Before writing the resolver, add 10–15 questions that name a species by an alternate name: GBIF/IUCN/Wikipedia names from `aliases.csv`, older genus placements, "Common"/"Eurasian"/"European" variants, US vs. UK names. Three questions are too few to show anything. Mark about a third of the eval set as **held out** (`split: holdout` in `questions.jsonl`): don't look at those while tuning on days 4–6, and report them separately on day 5, so the numbers aren't just fitted to the questions used for tuning.
   - ✅ **Done:** 20 questions added (q20–q39), 39 in all. **18 alias questions:** the 3 from day 3 plus 15 new, covering UK/US names (Red-throated Diver, Shore Lark, Whistling Swan, Lapland Bunting, Black-necked Grebe), older English names (Hedge Sparrow, Eurasian Stone-Curlew), GBIF/IUCN names (Eurasian Black Vulture, Eastern Imperial Eagle, Marbled Teal, White-eyed Pochard), a prefix variant (Eurasian Marsh Harrier), scientific names (*Otis tarda*, *Syrrhaptes paradoxus*) and an older genus (*Chettusia gregaria*). Two `everyday_word` traps: "How **swift** is the Peregrine Falcon" and the out-of-scope Bald **Eagle**. **11 held out (28%):** 7 alias, 2 standard-name and 2 out-of-scope (Bald Eagle, Red-crowned Crane). The questions from day 3 stay in the tuning split, since their results have already been seen.
   - `scripts/eval_retrieval.py` scores held-out questions in its totals but doesn't list them one by one unless `--show-holdout` is given. The loader validates `split`, and tests check the alias counts and the held-out share.
   - **New baseline** (semantic only, top 6):

     | Questions | n | hit@1 | hit@6 | MRR | section hit@6 |
     |---|---|---|---|---|---|
     | Tune: standard names | 13 | 100% | 100% | 1.00 | 92% |
     | Tune: alternate names | 11 | 36% | 64% | 0.47 | 36% |
     | Holdout: standard names | 2 | 100% | 100% | 1.00 | 100% |
     | Holdout: alternate names | 7 | 57% | 71% | 0.62 | 57% |
     | All in-scope | 33 | 70% | 82% | 0.74 | 67% |

     Misses in the tuning set: goatsucker, *Falco cherrug*, Hedge Sparrow (House Sparrow wins), *Chettusia gregaria*. The live test (`test_answer_live.py --live`) reads the same file, so it now asks 39 questions and should fail on these until name resolution lands.
2. **Name resolution in the query path** (`resolve.py`). Before retrieval, normalize the query with the same function the alias index uses, then match the **longest** n-gram first so "eurasian eagle owl" isn't read as "eagle owl" or "owl."
   - **Everyday words:** multi-word aliases and scientific names always count. A one-word alias counts only if it isn't on a reviewed stoplist of one-word aliases that are also ordinary English ("swift", "kite", "rail", "crane", "tit", "shag", ...). Build the list from the one-word aliases in `aliases.csv` and check it by hand. "How swift is the peregrine falcon" must resolve to the Peregrine only.
   - **One species:** resolve to the canonical ID and filter retrieval to it. "What does the common goatsucker eat" now pulls the nightjar's chunks. If the filtered search comes back empty, fall back to unfiltered.
   - **Several species named** ("difference between the saker and the peregrine"): filter to all of them.
   - **A specific alias with 2–4 candidates:** the bot asks which bird the user means, with one inline button per candidate, then answers for the one picked. It never answers for each candidate, because that multiplies the model cost.
   - **A group alias** ("warbler", "eagle"), **or any alias with more than 4 candidates:** treat it as a topic, not something to resolve. No buttons and no species filter; retrieval runs unfiltered, optionally boosting the group's species. A wall of 30 buttons is unusable.
   - **No match:** retrieval runs as on day 3, and the day-3 decline messages apply. Now that names are resolved against the species list, consider making the `bird_not_found` wording firmer when the query clearly names a bird that isn't on the list.
   - Surface the resolved canonical name in the answer ("Common Nightjar (also: common goatsucker)") so the user sees the link.
   - ✅ **Done** (`resolve.py`, wired into `rag.answer`, the bot, the Streamlit app and `eval_retrieval.py`). As built, where it differs from or adds to the above:
     - **The everyday-word list has two rules** (`data/manual/everyday_words.csv`, each word with a note). Reviewing the 421 one-word aliases turned up two kinds of problem, not one:
       - `never`: junk that is never a bird name. These were bad Wikipedia redirects ("little" → Little Tern, "egypt" → Spotted Flycatcher, "booby" → Common Tern), name fragments ("bearded", "snowy"), folk names that are ordinary words ("snowflake", "whistler") and abbreviations ("blk", "thr").
       - `alone` (30 words): real bird names that are also ordinary words ("swift", "hobby", "merlin", "robin", "crane", "dove"). These count only when the question names no other bird, so "How swift is the Peregrine Falcon" is about the Peregrine alone and "Where does the merlin breed?" is still about the Merlin.
       - A test checks that every listed word really is a one-word alias.
       - ✅ **Junk removed at the source (2026-10-16):** 12 aliases that aren't names for their bird now have `remove` rows in `alias_overrides.csv`. They are fragments ("Bearded", "Purple", "Snowy", "Little"), lone surnames ("Bewick's", "Cetti's", "Temminck's"), abbreviations ("Blk", "Thr") and wrong pairings ("Egypt", "Booby", "Kāhawai").
         - The dumps were rebuilt offline from the raw cache. The diff is exactly those 12 alias rows plus the same names in 12 species' "Also known as" lines; the IUCN snapshot is byte-identical. The index was rebuilt to match.
         - Those 12 words are off the everyday-word list, which keeps 14 `never` words. These are genuine folk names that are ordinary words ("Billy" for the Dunnock, "Seahawk" for the Osprey): they stay in the alias table and are only blocked at query time. The resulting list is 44 words.
         - ✅ Docker image rebuilt (2026-10-18; the dependency and model layers came from cache). It passed the corpus-fingerprint check at build time. A smoke test with no network and no keys showed name resolution working inside the container: "common goatsucker" → Eurasian Nightjar; "How swift is the Peregrine Falcon" → Peregrine only; "little birds" → no species; "cat owl" → ambiguous between Long-eared and Tawny Owl. Without a token, the bot still exits with a clear message.
     - **Group vs. ambiguous:** only 4 aliases map to 2+ species without being group names ("cat owl", "pond bird", "northern grey shrike", "red rumped swallow"), each with 2 candidates, so buttons will be rare. Only the first ambiguous name in a question is asked about.
     - **Several species:** each species gets an equal share of the k places (ceiling of k/n), merged by score. That way a comparison question sees every bird it names, not just the one that happens to match best. Group names in the same question don't filter.
     - **The model is told about renamed birds:** a `<names>` block before the question says `"Common goatsucker" in the question means the Eurasian Nightjar.` Without it, the model could decline because no source mentions the name the user used. This changes the prompt, so ⬜ re-run `.venv/bin/pytest tests/test_answer_live.py --live`.
     - **Display:** the Telegram reply starts with *Eurasian Nightjar (also: Common goatsucker)*. The display spelling prefers eBird, then IUCN, then GBIF, with Wikipedia redirects last ("The Black-necked grebe").
     - **Choice buttons** carry `c:<question key>:<candidate number>`. The answer for the chosen bird is cached under a key that includes the choice, so "Show passages" works on it. An expired question asks the user to ask again.
     - **The `bird_not_found` wording stays hedged.** A misspelled name ("Sakker Falcon") matches no alias either, so "no alias matched" can't tell a bird that isn't covered from a typo.
   - **Result** (retrieval eval, top 6): **100% hit@1 on all 33 in-scope questions**, up from 70%. Alternate names went from 36% to 100% on the tuning split and from 57% to 100% on the held-out split. Section hit@6 went from 67% to 94%; the misses are q23 (Black Vulture status) and q28 (Peregrine speed). `--no-names` reproduces the old baseline.
     - **Caveat for day 5:** once a name resolves, the filter all but guarantees the right species, so species-level hit@k now mostly measures the resolver. **Section hit@k** becomes the informative retrieval metric, and the held-out alias questions are the real test of the resolver.
   - **Live run after name resolution (2026-10-18): 37 of 39 passed.** The two failures and their fixes:
     - **q23** ("Why is the Eurasian Black Vulture threatened?"): retrieval failure. The typed name literally appears in the bird's Introduction ("also known as … Eurasian black vulture"), which pushed the Status passage to rank 7. ✅ **Fix:** search with the canonical name in place of the typed one (`rag.search_text`; the resolver now records each match's span in the question). The model still sees the question as asked, plus the `<names>` note. Result: the Status passage is at rank 2. Section hit@6 is 100% on the tuning alias questions (was 91%) and 97% overall (was 94%); q15, q24 and q26 now have the right section at rank 1.
     - **q38** ("What does the Bald Eagle eat?"): the model answered, with correct citations, from the White-tailed Eagle's diet section, which compares the two eagles. That breaks the scope rule, though it isn't a hallucination. ✅ **Fix (a prompt rule, chosen over a world-species list):** a source is about a bird only if its `species` attribute names it, and a passing mention doesn't count. A question about a covered bird that mentions an uncovered one is answered about the covered one. If the model gets it wrong anyway, the citations show which bird the text is about.
     - **Holdout bookkeeping:** q38 was used to make that fix, so it moved to the tuning split. **q40** ("What does the American Kestrel eat?") replaces it in the holdout split: same trap, since four covered birds' articles mention it. The set is now 40 questions, 11 held out (28%).
     - ✅ Live re-run after both fixes (2026-10-18): **all 40 questions passed**, including q23, q38 and the new q40; no standard-name question was wrongly declined under the new rule. The container was restarted on the rebuilt image.
3. **Query router**, a small keyword/rule classifier (document the logic), with three paths. A status or live question that names no species gets asked which bird it's about.
   1. **Status** ("conservation status of X," "is X endangered"): answer from the IUCN snapshot with its category, assessment year and fetch date. Handle the gaps: if the snapshot file is missing (a fresh clone without an IUCN token, or a container without `data/private/` mounted), say status is unavailable and still answer other questions. For the 23 species with no IUCN record, say so rather than guess. ⬜ Check that IUCN's terms allow *showing* a category to bot users (displaying it, as distinct from redistributing the data), and record the answer in `DATA_LICENSES.md`.
   2. **Live** ("has X been seen recently," "where can I see X in Kazakhstan now"): call the eBird recent-observations endpoint for `KZ` and the species code, with a timeout and a graceful failure message. **eBird withholds sensitive species** from this endpoint, and the notable species (Saker Falcon, Sociable Lapwing, ...) are among the likeliest to be withheld. An empty result therefore reads "no recent public reports on eBird", never "not seen". Credit eBird as the source.
   3. **Descriptive:** retrieval over the corpus (semantic, with name resolution; hybrid only if step 5 is ever built).
   - ✅ **Done** (`router.py`, wired into `rag.answer`). As built:
     - **Classifier:** a few regular expressions, deliberately narrow; anything unclear stays descriptive. Live is checked first: "seen/spotted/reported … recently/lately/now", "recent/latest/last/any sightings", "where can I see … now". Status: "conservation status", "Red List", "IUCN", "status of", "is/are … endangered/threatened/vulnerable/extinct/at risk", "how endangered/threatened". **Questions asking for reasons stay descriptive** ("why", "because", "threats", "causes", "declin…"): "Why is the Eurasian Black Vulture threatened?" is answered from the article, since a category explains nothing. A test checks that all 40 eval questions stay descriptive, so routing can't quietly divert the eval.
     - **Status and live answers are written from the data, with no retrieval or model call.** Each cites its source as `[1]`, so the bot links it and lists it like any other answer (labels "IUCN Red List" and "eBird recent sightings"; "Show passages" leaves out the similarity score for them).
     - **Status answer:** category in full and as a code, year of the global assessment, a link to the assessment, and "Red List data retrieved <date>". For the 23 species without an assessment, it says IUCN has none under the names checked. If the snapshot is missing, it says status isn't available on this server. The snapshot is read on each status question, so mounting it later works without a restart.
     - **Live answer:** asks eBird for the last 30 days (its maximum). eBird returns the latest sighting per location, so the answer says "reports from N locations" and lists up to 5, newest first, with counts. eBird's private locations show as "a private location". Empty results get the sparse-coverage and sensitive-species wording. Errors and a missing key get "couldn't reach eBird, try again".
     - **Live calls use their own eBird client** (`ebird_client._get_live_client`): 8 s timeout and one retry, instead of the batch client's three retries with backoff.
     - **A status or live question that names no covered bird** ("Are eagles endangered?", or a bird not on the list) asks the user to name a Kazakhstan bird, with an example.
     - The route is on the `Answer` and in the log line (`route=status`); the new decline reasons are `species_needed`, `status_unavailable` and `live_unavailable`.
     - **Docker:** status answers need the snapshot mount, now part of the documented `docker run` command (`-v "$PWD/data/private:/app/data/private:ro"`). `EBIRD_API_KEY` is now used at run time, and `.env.example` says so.
     - **IUCN terms (the ⬜ above):** checked, but no definite answer. The official terms page refused automated access, and the secondary sources found quote a ban on redistribution "by any means". Showing categories in today's private bot is personal use; ⚠ settle this with IUCN before opening the bot to the public. Recorded in `DATA_LICENSES.md`.
4. **Make the answer cache route-aware.** Key it on the normalized question (the same normalization, so case and spacing don't create misses). Live-path answers aren't cached, since "seen recently" goes stale.
   - ✅ **Done** (`bot.AnswerCache`, `bot.normalize_question`). As built, with two departures from the line above:
     - **Live answers are kept for 10 minutes** (`LIVE_CACHE_SECONDS`), not left out entirely. The "Show passages" button finds its answer through the cache, so with no entry it would always say "expired". Ten minutes covers the button and a quick repeat, and the next ask after that fetches fresh sightings.
     - **The key normalization is not `aliases.normalize`:** that drops non-Latin letters, so every question typed in Cyrillic would share one key. The key instead ignores case, punctuation (any script) and spacing ("What does it eat?" = "what does it eat").
     - **Temporary failures are never cached** ("couldn't reach eBird", "IUCN snapshot not available"), so the next ask tries again; a failure also evicts an older answer to the same question. Status and descriptive answers are cached as before.
     - Tests use a fake clock: a live answer expires at 600 s, while status and descriptive answers stay; failures aren't stored; a live question asked at 0, 5 and 11 minutes is fetched twice.
     - Image rebuilt. Smoke test in the container (no network, no keys): with `data/private` mounted, a status question is answered from the snapshot; without the mount, it gets the "not available" reply; a live question gets "couldn't reach eBird", as expected without network or key.
     - ✅ Hand test against real Telegram (2026-10-18), container restarted with the snapshot mount: status answered from the snapshot; live answered by eBird in about 0.6 s (so the key works); "Are eagles endangered?" got `species_needed`; a repeated status question came from the cache; and a live question repeated after 4½ minutes came from the cache (`route=live cached=True`). No errors in the logs.
5. **Optional (not implemented): BM25 keyword index.** Bookmarked on 2026-10-18 to add later if wanted. **Why it waits:** name resolution already closed the exact-name gap BM25 was meant for. After steps 2–4, 100% of in-scope eval questions get the right bird first and 97% get the right section in the top 6, so this eval set leaves BM25 almost nothing to show. Worth revisiting if the eval grows questions that hinge on rare exact terms (place names, prey species, measurements) rather than bird names. As originally planned: a BM25 keyword index (`rank-bm25`) over the same chunks, fused with semantic results by reciprocal rank fusion; make the blend a config knob. **Tokenization matters:** handle hyphenated and multi-word names ("Black-capped," "Pallas's Sandgrouse") with the same normalization as the alias index. 

**Test**
- Retrieval eval (`scripts/eval_retrieval.py`): semantic baseline vs. + name resolution (+ hybrid, if step 5 is built), with alias questions and held-out questions reported as separate rows. Report the numbers whichever way they go.
- Name resolution unit tests: every alternate name of a species resolves to the same ID; the longest match wins; a 2–4-candidate alias produces buttons; a group alias produces no filter; an unknown name passes through; everyday-word negatives ("swift", "rail", ...) don't resolve.
- Router unit tests: each query type lands on the right path; eBird failure and eBird empty results (the exact wording); missing IUCN snapshot; a species with no IUCN record.
- Cache: a live-path answer is never served from the cache.
- Regression: `.venv/bin/pytest tests/test_answer_live.py --live` still passes the day-3 questions, and q13/q15 (goatsucker, *Falco cherrug*) now pass.

**End of day:** alternate names resolve to the right bird, every question type lands on the right path (descriptive, status or live), and the live eBird path is part of the query flow, each step with its own retrieval numbers.

---

## Day 5 — Evaluation harness  *(the thing most portfolios skip)*

**Implement**
- Grow the set to **~40 questions** (day 4 already adds the alias questions) and write a **reference answer** for each. The current questions have none, and writing them is most of the day's work, so budget the morning for it. Keep the held-out split from day 4. Label ground truth at the **species level** (which species' chunks should be retrieved, optionally plus section); chunk-level labels are too expensive to hand-build and break on every re-chunk. Cover all query types, and **include several pairs that ask about a species by an alternate English name**, plus a few out-of-scope questions where the correct behavior is declining.
  - ✅ **Size and routed questions (2026-10-19).** The set was already at 40 after day 4. The gap was routing: no question went through the router, so a "+ routing" row would measure nothing. **6 routed questions added (q41–q46), 46 in all:**
    - Status: Saker Falcon (EN), Sociable *Plover* (alias → CR), and Swan Goose (no IUCN assessment: the answer must say so).
    - Live: Demoiselle Crane, and Hedge *Sparrow* (alias → Dunnock).
    - "Are owls endangered?": names no single bird, so the bot should ask which.
    - **Held out: 13 of 46 (28%)**, including 2 of the routed questions.
    - **Format:** a new optional `route` field (`descriptive` by default, `status`, `live`) and a new type `sightings`. A routed question has no expected sections, and is "declined" exactly when it names no bird (the bot asks for one). Routed questions are left out of the retrieval metrics, since nothing is retrieved for them.
    - **Routing check** (`evaluation.evaluate_routing`, free): each question must take its expected path and, if routed, look up exactly its expected birds. `scripts/eval_retrieval.py` prints it ("46 of 46 take their expected path"), and a test enforces it. That test replaces the day-4 test that kept every question descriptive.
    - **Live test:** routed questions get their own checks: the right path, a citation to the right lookup (IUCN or eBird) for the right bird, the "no assessment" wording for the Swan Goose, and `species_needed` for the owls. They make no model call; status needs the private snapshot (skipped without it) and live needs `EBIRD_API_KEY`. ✅ Run `.venv/bin/pytest tests/test_answer_live.py --live -k routed -v` (free apart from eBird calls).
  - ✅ **Reference answers (2026-10-19):** every one of the 46 questions has a `reference_answer`.
    - **Descriptive (33):** a 1–3 sentence paraphrase of the facts in the question's expected sections, written from the corpus text, not from general knowledge, and never copying its wording, since the eval file is published with the code.
    - **Alternate-name questions:** the reference also says which bird it is ("The common goatsucker is the Eurasian Nightjar, and the answer should say so"), because a good answer should.
    - **Out of scope (7):** the expected decline and its reason (`bird_not_found` or `not_about_birds`), plus what must *not* be used (for example, other cranes' text for the Black-necked Crane).
    - **Routed (6):** the expected behavior. **Status references name no category,** since the category is IUCN data. Live references describe the two acceptable forms, because sightings change daily.
    - **Judgment calls to review:** q28 (Peregrine speed) expects the answer to say the sources disagree (claims of 320–389 km/h vs. 184 km/h reliably measured), not just quote the largest number. q09 expects "causes largely unknown; hunting a notable threat". Several references (q04, q10, q23) list more points than a short answer will cover, so the rubric should ask whether the main points are there, not all of them.
    - **Tests:** every question has a reference; alias references name the canonical bird; status references contain no Red List category.
    - ✅ Read the references once; they were drafted by Claude from the corpus.
- Retrieval metrics: hit-rate / recall@k and MRR at the species level.
- Answer metrics: a **documented manual rubric** (faithful / relevant / properly cited / correctly declined) is the default. Add RAGAS faithfulness + answer-relevance as an optional second column if time allows. With ~30 questions, LLM-judge scores are noisy, so report them with that caveat.
  - ✅ **Rubric and grading tools (2026-10-19).**
    - **The rubric** is `eval/RUBRIC.md`. Four mechanical checks are filled in automatically:
      - `behavior_ok`: answered, declined with the right reason, or asked for a bird, as the question calls for;
      - `route_ok`;
      - `citations_valid`: no invented `[n]`;
      - `right_bird`: a cited source is about the expected species.
    - **Three human scores, 0–2**, given only to descriptive questions the bot rightly answered (33 of 46): **faithful** (every claim supported by the passages it cites, judged against the passages, not the reference), **complete** (the reference's *main* points, not every detail), and **cited** (every factual sentence cited, each citation supporting its sentence). Declines, wrong behavior and routed answers (written from data) are pre-filled `n/a`.
    - **`scripts/answer_eval.py`** asks all 46 questions (40 model calls) and writes `data/eval_runs/<name>/`:
      - `answers.jsonl`: the full records;
      - `sheet.md`: each question, its reference, the answer and the cited passages, for reading while grading;
      - `grades.csv`: checks filled in, scores blank.
      Runs are under `data/` because the sheet quotes Wikipedia text.
    - **`scripts/score_eval.py`** prints pass rates and mean scores (with the share scoring 2) for all questions, tune, holdout, standard and alternate names, and lists every failed check.
    - The expected decline reason is read from the reference's `Decline (<reason>)` prefix; a test checks that every out-of-scope reference has one.
    - Tests in `tests/test_grading.py`. A dry run on fake answers confirmed the table.
    - **Ablation note:** grading by hand three configurations × 33 answers is too much. The **retrieval** ablation (semantic → + names) is automatic (`eval_retrieval.py --no-names`), and the hand-graded **answer** metrics are for the full system.
    - ⬜ Run `scripts/answer_eval.py --name baseline`, grade `grades.csv`, then `scripts/score_eval.py`.
- A script that runs the whole eval and dumps a results table (semantic-only vs. + name resolution vs. + routing; + hybrid only if step 5 is built), with held-out questions as a separate row.

**Test**
- The eval runs end-to-end and reports numbers.
- Confirm whether the ablation shows the expected story (name resolution > semantic on alternate-name queries). If it doesn't, that's a finding; report it honestly.

**End of day:** quantitative evidence the system works, with an ablation.

---

## Day 6 — Cross-encoder reranker + buffer

This day is **explicitly the buffer**: if days 1–5 overran, spend it finishing them and skip or shrink the reranker. BM25 (day 4, step 5) is an option for spare time here, not a must.

**Implement (stretch)**
- Add a reranking stage: retrieve a wider candidate set (say the top 20), re-score with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), and keep the top 6 for the prompt (the same count as day 3, so the ablation compares like with like).
- Make it a config flag so I can run the eval with and without it.
- If RAGAS was deferred from day 5, add it here.

**Test**
- Re-run the day-5 eval with reranking on; compare recall@k and the answer-quality rubric against the day-5 numbers. Add the row to the ablation table.
- Confirm latency is still acceptable for a chat reply (a few seconds) (rerank a bounded candidate set, not the whole corpus).

**End of day:** a resolve → retrieve → rerank pipeline with numbers showing what each stage contributes, or a documented decision to skip it.

---

## Day 7 — Polish, README, write-up

**Implement**
- README: problem (Kazakhstan birds and the multi-name problem), architecture diagram, data sources, how to run (including creating a bot with @BotFather and setting the allowlist), **example questions with screenshots of the Telegram chat**, and **my eval numbers + ablation table**. Recruiters read the README and the eval — put them first.
- In the architecture diagram, show the **dump-vs-live split** (offline `build_dumps.py` → committed artifacts → app; eBird live in the query path) and say in one line *why* each source sits where it does.
- Clean the repo: `.env.example`, pinned `requirements.txt`, module structure, docstrings, `DATA_LICENSES.md`.
- **Publish the source without the data.** `data/dumps/` (including the Wikipedia corpus) is tracked in the local git history, so pushing this repo as it is would publish the data. Publish from a history that never contained `data/dumps/`: a fresh repo with a source-only first commit, or a filtered copy of this one. Keep this repo as the private working copy. Check the published tree and its history for anything under `data/` before the first push.
- Short "design decisions & limitations" section: scope bounds (the eBird `KZ` list, and why frequency ranking is off for sparse regions), canonical-ID choice, ambiguity policy, why BM25 was left optional, why the data isn't published, small eval-set caveat, and what I'd do with more time (Russian/Kazakh names, multi-turn chat).
- Record a 60–90s screen capture of a conversation with the bot.

**Test**
- Fresh-clone test of the **published** repo: follow my own README from scratch, including regenerating the data with fresh keys (`build_dumps.py`, then `build_index.py`), and confirm the bot runs.
- Final pass over the eval table so the numbers match what the README claims.

**End of day:** shippable, and it tells a clear story: applied LLM + RAG, with rigor.

---

## Scope discipline
- **Non-negotiable core (days 1–3):** working RAG + demo over the fixed Kazakhstan species list, with the alias index (including its ambiguity handling) built from day 1, and the seed eval file started.
- **Differentiators (days 4–5):** name resolution + routing (including live eBird) + eval. (Hybrid BM25 is optional.) These set the project apart from tutorial projects — protect them.
- **Stretch and buffer (day 6):** the reranker. Cut first if time runs short — the eval and ablation matter more than one extra pipeline stage.
- Keep species scope small and fixed. Breadth is not what's being evaluated; engineering judgment is.
