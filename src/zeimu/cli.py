"""Command-line entry point: ``uv run zeimu <command>``."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

import httpx

from zeimu.index.embedder import Embedder, SentenceTransformerEmbedder
from zeimu.index.retriever import BM25Retriever, EmbeddingRetriever, Retriever
from zeimu.index.tokenizer import SudachiTokenizer
from zeimu.index.vector_cache import StaleCacheError, VectorCache
from zeimu.ingest.cache import RawPageCache
from zeimu.ingest.chunker import DEFAULT_MAX_CHARS
from zeimu.ingest.corpus import build_chunks, read_jsonl, write_jsonl
from zeimu.ingest.discover import (
    CODE_INDEX_URL,
    CORPUS_CATEGORIES,
    NTA_HOST,
    find_tax_answer_links,
)
from zeimu.ingest.fetcher import FetchError, PoliteFetcher

USER_AGENT = "zeimu/0.1 (+https://github.com/realkyle/zeimu)"
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_CHUNKS_PATH = Path("data/chunks.jsonl")
DEFAULT_EMBEDDINGS_DIR = Path("data/embeddings")

logger = logging.getLogger("zeimu")


def build_fetcher(client: httpx.Client, raw_dir: Path, min_interval: float) -> PoliteFetcher:
    """Wire up the production fetcher for the NTA site."""
    return PoliteFetcher(
        client,
        RawPageCache(raw_dir),
        user_agent=USER_AGENT,
        allowed_hosts=frozenset({NTA_HOST}),
        min_interval=min_interval,
    )


def fetch_corpus(fetcher: PoliteFetcher, *, refresh: bool) -> tuple[int, list[str]]:
    """Fetch the code index, every corpus page and the Q&A pages they link to.

    Returns the number of pages now cached and the URLs that failed. A failed page is
    logged and skipped so one bad link doesn't lose the rest of the run.
    """
    index = fetcher.fetch(CODE_INDEX_URL, refresh=refresh)
    main_urls = find_tax_answer_links(
        index.body.decode("utf-8"), CODE_INDEX_URL, categories=CORPUS_CATEGORIES, kind="main"
    )
    failed: list[str] = []
    qa_urls: set[str] = set()
    ok = 0
    for url in main_urls:
        try:
            page = fetcher.fetch(url, refresh=refresh)
        except FetchError as error:
            logger.warning("skipped: %s", error)
            failed.append(url)
            continue
        ok += 1
        qa_urls.update(
            find_tax_answer_links(
                page.body.decode("utf-8"), url, categories=CORPUS_CATEGORIES, kind="qa"
            )
        )
    for url in sorted(qa_urls):
        try:
            fetcher.fetch(url, refresh=refresh)
            ok += 1
        except FetchError as error:
            logger.warning("skipped: %s", error)
            failed.append(url)
    logger.info(
        "%d main pages, %d Q&A pages, %d cached OK, %d failed, %d network requests",
        len(main_urls),
        len(qa_urls),
        ok,
        len(failed),
        fetcher.network_requests,
    )
    return ok, failed


def chunk_corpus(raw_dir: Path, out: Path, max_chars: int) -> int:
    """Clean and chunk every cached page into ``out``. Returns the number of warnings."""
    chunks, report = build_chunks(RawPageCache(raw_dir).iter_pages(), max_chars=max_chars)
    write_jsonl(chunks, out)
    for warning in report.warnings:
        logger.warning(warning)
    sizes = sorted(len(chunk.text) for chunk in chunks)
    logger.info(
        "%d pages -> %d chunks in %s (chars: median %d, max %d), %d warnings",
        report.pages,
        report.chunks,
        out,
        sizes[len(sizes) // 2] if sizes else 0,
        sizes[-1] if sizes else 0,
        len(report.warnings),
    )
    return len(report.warnings)


def embed_corpus(chunks_path: Path, cache_dir: Path, embedder: Embedder) -> int:
    """Embed every chunk and save the vectors to ``cache_dir``. Returns the chunk count."""
    chunks = read_jsonl(chunks_path)
    EmbeddingRetriever.build(chunks, embedder, VectorCache(cache_dir))
    logger.info("%d chunks embedded with %s -> %s", len(chunks), embedder.model_id, cache_dir)
    return len(chunks)


def search_corpus(
    chunks_path: Path,
    query: str,
    k: int,
    *,
    method: str = "bm25",
    cache_dir: Path = DEFAULT_EMBEDDINGS_DIR,
    embedder: Embedder | None = None,
) -> list[str]:
    """Search by ``method`` ("bm25" or "embed"); return lines of rank, score, ID, heading.

    Embedding search reads vectors saved by ``zeimu embed``; a missing or stale cache raises
    ``FileNotFoundError`` / ``StaleCacheError`` rather than silently re-embedding.
    """
    chunks = read_jsonl(chunks_path)
    by_id = {chunk.id: chunk for chunk in chunks}
    retriever: Retriever
    if method == "bm25":
        retriever = BM25Retriever(chunks, SudachiTokenizer())
    elif method == "embed":
        model = embedder if embedder is not None else SentenceTransformerEmbedder()
        retriever = EmbeddingRetriever.from_cache(chunks, model, VectorCache(cache_dir))
    else:
        raise ValueError(f"unknown search method {method!r}")
    lines: list[str] = []
    for hit in retriever.search(query, k=k):
        chunk = by_id[hit.chunk_id]
        heading = " > ".join((chunk.title, *chunk.heading_path))
        lines.append(f"{hit.rank:>2}. {hit.score:6.2f}  {hit.chunk_id:<22} {heading}")
    return lines or ["(no matches)"]


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the chosen command. Returns the process exit code."""
    parser = argparse.ArgumentParser(prog="zeimu")
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch", help="fetch NTA Tax Answer pages into the raw cache")
    fetch.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    fetch.add_argument("--min-interval", type=float, default=1.5, help="seconds between requests")
    fetch.add_argument("--refresh", action="store_true", help="re-fetch pages already cached")
    chunk = commands.add_parser("chunk", help="clean and chunk cached pages into JSONL")
    chunk.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    chunk.add_argument("--out", type=Path, default=DEFAULT_CHUNKS_PATH)
    chunk.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    embed = commands.add_parser("embed", help="embed the chunks with bge-m3 into a vector cache")
    embed.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    embed.add_argument("--out", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    search = commands.add_parser("search", help="search the chunks (BM25 or embeddings)")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5, help="number of results")
    search.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    search.add_argument("--method", choices=("bm25", "embed"), default="bm25")
    search.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS_DIR)
    args = parser.parse_args(argv)

    if args.command == "search":
        try:
            lines = search_corpus(
                args.chunks, args.query, args.k, method=args.method, cache_dir=args.embeddings
            )
        except (FileNotFoundError, StaleCacheError) as error:
            print(f"zeimu: {error}. Run `zeimu embed` to (re)build the vector cache.")
            return 1
        print("\n".join(lines))
        return 0

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    # Per-request INFO lines: our fetcher already logs each GET, and model downloads are noise.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if args.command == "embed":
        embed_corpus(args.chunks, args.out, SentenceTransformerEmbedder())
        return 0
    if args.command == "chunk":
        return 1 if chunk_corpus(args.raw_dir, args.out, args.max_chars) else 0

    with httpx.Client(timeout=30.0) as client:
        fetcher = build_fetcher(client, args.raw_dir, args.min_interval)
        _, failed = fetch_corpus(fetcher, refresh=args.refresh)
    return 1 if failed else 0
