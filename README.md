# zeimu (税務)

An English-first assistant for **Japanese inheritance and gift tax** (相続税・贈与税) that answers from official National Tax Agency (国税庁) sources.

> **Not tax advice.** zeimu is a personal engineering project. Its answers may be wrong or out of date. Consult a licensed tax accountant (税理士) before making any decision.

## Goal

Answer questions in English or Japanese with:

- **Citations** to the exact source section for every claim
- **Exact calculations** from a deterministic calculator tool; the language model never does arithmetic
- **Refusals** when the sources don't support an answer
- **A measured, harm-weighted evaluation** of how often it is right, cites correctly and refuses correctly

## Status

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold: uv, ruff, Pyrefly, pytest, pre-commit, CI | In progress |
| 1 | Corpus and hybrid retrieval (BM25 + embeddings) | Not started |
| 2 | Cited answers, citation verification, refusals | Not started |
| 3 | Calculator tool and agent loop | Not started |
| 4 | Robustness: prompt injection, EN/JA consistency, date-dependent rules | Not started |
| 5 | API, web UI, deployment, CI evaluation gate | Not started |

No evaluation results yet. Numbers will appear here only once they have been measured.

## Development

Requires [uv](https://docs.astral.sh/uv/). Python 3.12 is installed by uv automatically.

```bash
uv sync                          # install dependencies
uv run pre-commit install        # run checks on every commit
uv run pytest                    # tests
uv run ruff check . && uv run ruff format --check .   # lint and format
uv run pyrefly check             # type check
```

## Sources and attribution

The corpus is built from the National Tax Agency's Tax Answer (タックスアンサー) pages, which may be reused under the Japanese government standard terms of use with attribution. zeimu **processes** this content (it splits, indexes, translates and summarizes it), so it is not the government's own wording:

> 国税庁「タックスアンサー」(https://www.nta.go.jp/taxes/shiraberu/taxanswer/) を加工して作成

Each answer links to the specific Tax Answer page it cites. No real client data is used; all examples and test cases use synthetic people and amounts.
