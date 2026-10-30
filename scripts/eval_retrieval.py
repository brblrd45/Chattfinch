"""Measure retrieval on the eval set (eval/questions.jsonl). Free: no model calls.

    .venv/bin/python scripts/eval_retrieval.py            # top 6 chunks, the app's setting
    .venv/bin/python scripts/eval_retrieval.py -k 20
    .venv/bin/python scripts/eval_retrieval.py --show-holdout
    .venv/bin/python scripts/eval_retrieval.py --no-names     # semantic search only

By default retrieval runs as the bot runs it: bird names in the question are
resolved first (resolve.py) and the search is filtered to the species they name.
--no-names turns that off, for the day-3 baseline.

For each in-scope question it reports where the first chunk from the expected
species ranks, and whether a chunk from the expected section was retrieved. Run it
before and after a retrieval change (day 4) to see what the change bought.

Held-out questions (split "holdout") are scored in the totals but their rows are
hidden unless --show-holdout is given: tune on the other questions, and use the
held-out totals to check that a change generalizes rather than fits the tuning set.
"""

import argparse

from chattfinch import config, evaluation, rag, resolve, retrieval, router


def print_metrics(label: str, summary: evaluation.RetrievalSummary) -> None:
    if not summary.n:
        return
    print(f"  {label:32} {summary.n:>3}  {summary.hit_at_1:>6.0%} {summary.hit_at_k:>6.0%} "
          f"{summary.mrr:>6.2f} {summary.section_hit_at_k:>8.0%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("-k", type=int, default=config.RETRIEVAL_K,
                        help=f"chunks retrieved per question (default {config.RETRIEVAL_K})")
    parser.add_argument("--show-holdout", action="store_true",
                        help="also list the held-out questions one by one")
    parser.add_argument("--no-names", action="store_true",
                        help="don't resolve bird names first (semantic search only)")
    args = parser.parse_args()

    if args.no_names:
        retrieve = retrieval.retrieve
    else:
        def retrieve(query, k):  # as rag.answer does it
            resolution = resolve.resolve(query)
            return rag.search(rag.search_text(query, list(resolution.species_matches)), k,
                              resolution.species)

    questions = evaluation.load_questions()
    summary = evaluation.evaluate_retrieval(questions, retrieve, args.k)
    print(f"Retrieval: {'semantic only' if args.no_names else 'name resolution + semantic'}\n")
    shown = [r for r in summary.results if args.show_holdout or r.question.split != "holdout"]

    print(f"{'id':4} {'type':15} {'species':>7} {'section':>7}  question")
    for result in shown:
        q = result.question
        rank = lambda value: str(value) if value else "miss"
        flags = "".join(f" [{tag}]" for tag in q.tags)
        if q.split == "holdout":
            flags += " [holdout]"
        wrong = "" if result.species_rank == 1 else f"   (top hit: {result.top_species})"
        print(f"{q.id:4} {q.type:15} {rank(result.species_rank):>7} "
              f"{rank(result.section_rank):>7}  {q.question}{flags}{wrong}")
    hidden = len(summary.results) - len(shown)
    if hidden:
        print(f"({hidden} held-out questions not listed; --show-holdout lists them)")

    # Re-scoring subsets from the same results: no second round of retrieval.
    def subset(keep) -> evaluation.RetrievalSummary:
        return evaluation.summarize([r for r in summary.results if keep(r.question)], args.k)

    print(f"\nTop {summary.k} chunks. hit@1: the top chunk is from the right species; "
          f"hit@{summary.k}: any is; MRR: mean 1/rank;\nsection: a chunk from the right "
          f"species and section was retrieved.\n")
    print(f"  {'':32} {'n':>3}  {'hit@1':>6} {f'hit@{summary.k}':>6} {'MRR':>6} {'section':>8}")
    tune = lambda q: q.split == "tune"
    print_metrics("tune: standard names", subset(lambda q: tune(q) and "alias" not in q.tags))
    print_metrics("tune: alternate names", subset(lambda q: tune(q) and "alias" in q.tags))
    print_metrics("holdout: standard names",
                  subset(lambda q: not tune(q) and "alias" not in q.tags))
    print_metrics("holdout: alternate names", subset(lambda q: not tune(q) and "alias" in q.tags))
    print_metrics("all in-scope questions", summary)

    routing = evaluation.evaluate_routing(questions, router.classify, resolve.resolve)
    routed = [r for r in routing if r.question.route != router.DESCRIPTIVE]
    wrong = [r for r in routing if not (r.route_ok and r.species_ok)]
    print(f"\nRouting: {len(routing) - len(wrong)} of {len(routing)} questions take their "
          f"expected path ({len(routed)} routed to a status or live lookup, which "
          "retrieves nothing and isn't in the table above)")
    for r in wrong:
        hidden = r.question.split == "holdout" and not args.show_holdout
        print(f"  {'(held out)' if hidden else r.question.id}: expected {r.question.route} "
              f"{list(r.question.expected_species)}, got {r.route} {list(r.species)}")

    out_of_scope = [q for q in questions if not q.in_scope and q.route == router.DESCRIPTIVE
                    and (args.show_holdout or q.split != "holdout")]
    print(f"\n{len(out_of_scope)} out-of-scope questions (best similarity score; retrieval "
          "always returns something, so declining is the model's job):")
    for q in out_of_scope:
        top = retrieval.retrieve(q.question, k=1)[0]
        print(f"  {q.id}  {top['score']:.2f}  {q.question}   "
              f"(nearest: {top['metadata']['common_name']})")


if __name__ == "__main__":
    main()
