"""Ask one question from the command line and print the answer with its sources.

    .venv/bin/python scripts/ask.py "What does the Saker Falcon eat?"
    .venv/bin/python scripts/ask.py --show-text "Tell me about penguins."

For reading answers against their sources by hand (the faithfulness spot-check).
Needs the built index and Anthropic credentials; each run is one paid API call.
"""

import argparse
import sys
import textwrap

from chattfinch import citations, config, llm, rag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("question")
    parser.add_argument("-k", type=int, default=config.RETRIEVAL_K,
                        help=f"number of chunks to retrieve (default {config.RETRIEVAL_K})")
    parser.add_argument("--show-text", action="store_true",
                        help="print each source's full text, to check the answer against it")
    args = parser.parse_args()
    try:
        result = rag.answer(args.question, k=args.k)
    except (llm.LLMError, FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if result.declined:
        print(f"DECLINED (reason: {result.decline_reason or 'none given'})\n")
    print(textwrap.fill(result.text, width=100, replace_whitespace=False), "\n")
    invalid = citations.invalid_citations(result)
    if invalid:
        print(f"WARNING: cites {invalid}, which match no source the model was given.\n")

    def show(sources: list[rag.Source]) -> None:
        for source in sources:
            print(f"  [{source.number}] {citations.source_label(source)} | "
                  f"similarity {source.score:.2f}\n      {source.url}")
            if args.show_text:
                print(textwrap.indent(textwrap.fill(source.text, width=96), "      "), "\n")

    cited, uncited = citations.cited_sources(result), citations.uncited_sources(result)
    if cited:
        print("Sources cited:")
        show(cited)
    if uncited:
        print("Retrieved but not cited:")
        show(uncited)
    return 0


if __name__ == "__main__":
    sys.exit(main())
