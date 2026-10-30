# Answer grading rubric

How the bot's answers to the eval set (`eval/questions.jsonl`) are judged. Each
question has a `reference_answer` saying what a correct answer contains, written from
the corpus sections the answer should come from.

A run has two kinds of check:

- **Mechanical checks**, filled in by `scripts/answer_eval.py`. They need no judgment.
- **Human scores** for every descriptive question the bot actually answered. They need
  a reader, because a script can tell *that* an answer cites the right bird, not
  *whether* each sentence is supported by what it cites.

## Mechanical checks (filled in automatically)

| Check | Pass when |
|---|---|
| `behavior_ok` | The bot did what the question calls for: **answered** an in-scope question, **declined** an out-of-scope one, or **asked** which bird is meant when a status or live question names none. For a decline, the reason must match too (`bird_not_found` vs `not_about_birds`). |
| `route_ok` | The question took its expected path: descriptive (retrieval and the model), status (IUCN snapshot) or live (eBird). |
| `citations_valid` | Every `[n]` in the answer matches a source the bot was given. An invented citation fails. |
| `right_bird` | At least one cited source is about the expected species. Applies to answered in-scope questions only. |

A question whose `behavior_ok` fails has no human scores: the bot did the wrong thing,
and that is the result.

## Human scores (0, 1 or 2)

Grade only rows that have empty score cells. Rows the script pre-filled with `n/a`
are declines, questions where the bot behaved wrongly, or routed answers, which are
written from data and checked mechanically.

Read the question, the reference answer, the bot's answer and the cited passages in
`sheet.md`, then score three things.

### Faithful: is every claim supported by the passages it cites?

- **2:** Every factual claim is supported by the cited passages. Rewording is fine;
  facts are not added.
- **1:** A minor detail goes beyond the passages (a small embellishment, a
  generalization a little broader than the text), but nothing a reader would be
  misled by.
- **0:** A material claim is unsupported by, or contradicts, the passages. This
  includes facts the model "knew" but the passages don't say, and facts taken from a
  passage about a different bird.

Judge against the **passages**, not against the reference or your own knowledge: a
true fact the passages don't contain still scores 0 or 1.

### Complete: does it answer the question, as the reference does?

- **2:** It answers what was asked and covers the reference's **main points**. It
  needn't list every detail: the references for broad questions (threats, feeding
  methods) are longer than a good short answer.
- **1:** Partly: some main points are missing, or it answers a nearby question. Also
  use 1 when it rightly says the sources lack a detail the reference also lacks.
- **0:** It misses the point, is about the wrong bird, or omits what the reference says
  is essential. For example, q28 must say the sources disagree about the Peregrine's
  speed, and every alternate-name question must make clear which bird it is.

### Cited: are the citations right?

- **2:** Every factual sentence has a citation, and each citation points to a passage
  that actually says it.
- **1:** Some sentences are uncited, or a citation points to the wrong passage while
  the right one was also given.
- **0:** No citations, or most of them don't support their sentences.

## Notes column

Write a few words for any score below 2 ("adds wingspan not in [2]", "misses Africa
wintering"). The notes are what make a failed run actionable.

## Reading the results

`scripts/score_eval.py` reports, per split (tune / holdout) and for alternate-name
questions:

- behavior, route, citation and right-bird pass rates over all questions;
- for graded answers, the mean of each score and the share that scored 2.

With about 40 questions, one answer moves a percentage by 2–3 points: treat small
differences between runs as noise, and read the notes.

**Held-out questions** are graded like the others, but don't tune prompts or retrieval
against their notes. Doing so would turn them into tuning questions.
