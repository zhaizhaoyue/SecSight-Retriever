from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, List, Sequence


'''
PLEASE RUN THE FOLLOWING COMMAND TO RUN THE PIPELINE:(DON'T FORGET INPUT YOUR EMAIL)

python -m src.cli run ` 
--stages download,postprocess,parse,clean,index,chunk,embed ` 
--download-email YOUR EMAIL ` 
--download-outdir data/raw_reports ` 
--download-companies-csv data/companies.csv ` 
--download-limit-10k 1 ` 
--download-limit-10q 1 ` 
--download-sleep 0.5 ` 
--postprocess-base data/raw_reports/sec-edgar-filings ` 
--postprocess-out data/raw_reports/standard ` 
--parse-input data/raw_reports/standard ` 
--parse-output data/processed ` 
--clean-input data/processed ` 
--clean-output data/clean ` 
--index-input data/clean ` 
--index-output data/silver ` 
--chunk-input data/silver ` 
--chunk-output data/chunked ` 
--chunk-workers 4 ` 
--embed-input data/chunked ` 
--embed-output data/index ` 
--embed-model BAAI/bge-base-en-v1.5 ` 
--embed-use-title 
'''


STAGE_SEQUENCE: Sequence[str] = (
    "download",
    "postprocess",
    "parse",
    "clean",
    "index",
    "chunk",
    "embed",
)

DEFAULT_DOWNLOAD_OUTDIR = Path("data/raw_reports")
DEFAULT_COMPANIES_CSV = Path("data/companies.csv")
DEFAULT_POSTPROCESS_BASE = DEFAULT_DOWNLOAD_OUTDIR / "sec-edgar-filings"
DEFAULT_POSTPROCESS_OUT = Path("data/raw_reports/standard")
DEFAULT_PARSE_OUTPUT = Path("data/processed")
DEFAULT_CLEAN_OUTPUT = Path("data/clean")
DEFAULT_INDEX_OUTPUT = Path("data/silver")
DEFAULT_CHUNK_OUTPUT = Path("data/chunked")
DEFAULT_EMBED_OUTPUT = Path("data/index")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _comma_split(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_stage_list(raw: str | None) -> List[str]:
    if not raw or raw.lower() in {"", "all", "full"}:
        return list(STAGE_SEQUENCE)
    stages = [stage.strip().lower() for stage in raw.split(",") if stage.strip()]
    unknown = [s for s in stages if s not in STAGE_SEQUENCE]
    if unknown:
        valid = ", ".join(STAGE_SEQUENCE)
        raise SystemExit(f"Unknown stage(s): {', '.join(unknown)}. Valid options: {valid}.")
    return stages


def add_download_arguments(parser: argparse.ArgumentParser, *, alias: bool = False) -> None:
    flags_email = ["--download-email"]
    flags_company = ["--download-company-name"]
    flags_outdir = ["--download-outdir"]
    flags_companies = ["--download-companies-csv"]
    if alias:
        flags_email.append("--email")
        flags_company.append("--company-name")
        flags_outdir.append("--outdir")
        flags_companies.append("--companies-csv")
    parser.add_argument(*flags_email, dest="download_email", help="Contact email used for SEC downloads")
    parser.add_argument(*flags_company, dest="download_company_name", default="FinanceLLMAssistant",
                       help="Company name embedded in the SEC User-Agent header")
    parser.add_argument(*flags_outdir, dest="download_outdir", default=str(DEFAULT_DOWNLOAD_OUTDIR),
                       help="Destination directory for SEC downloads (default: data/raw_reports)")
    parser.add_argument(*flags_companies, dest="download_companies_csv", default=str(DEFAULT_COMPANIES_CSV),
                       help="CSV containing companies metadata (default: data/companies.csv)")
    parser.add_argument("--download-limit-10k", type=int, dest="download_limit_10k", default=None,
                       help="Download the latest N 10-K filings")
    parser.add_argument("--download-limit-10q", type=int, dest="download_limit_10q", default=None,
                       help="Download the latest N 10-Q filings")
    parser.add_argument("--download-no-xbrl", dest="download_no_xbrl", action="store_true",
                       help="Skip XBRL downloads")
    parser.add_argument("--download-sleep", type=float, dest="download_sleep", default=0.4,
                       help="Minimum delay between XBRL downloads in seconds (default: 0.4)")


def add_postprocess_arguments(parser: argparse.ArgumentParser, *, alias: bool = False) -> None:
    flags_base = ["--postprocess-base"]
    flags_out = ["--postprocess-out"]
    if alias:
        flags_base.append("--base-dir")
        flags_out.append("--output")
    parser.add_argument(*flags_base, dest="postprocess_base", default=str(DEFAULT_POSTPROCESS_BASE),
                       help="Input directory containing sec-edgar-filings")
    parser.add_argument(*flags_out, dest="postprocess_out", default=str(DEFAULT_POSTPROCESS_OUT),
                       help="Output directory for normalized filings (default: data/raw_reports/standard)")


def add_parse_arguments(parser: argparse.ArgumentParser, *, alias: bool = False) -> None:
    flags_in = ["--parse-input"]
    flags_out = ["--parse-output"]
    if alias:
        flags_in.append("--input")
        flags_out.append("--output")
    parser.add_argument(*flags_in, dest="parse_input", default=str(DEFAULT_POSTPROCESS_OUT),
                       help="Directory containing standard HTML/TXT crops (default: data/raw_reports/standard)")
    parser.add_argument(*flags_out, dest="parse_output", default=str(DEFAULT_PARSE_OUTPUT),
                       help="Directory for parsed text outputs (default: data/processed)")


def add_clean_arguments(parser: argparse.ArgumentParser, *, alias: bool = False) -> None:
    flags_in = ["--clean-input"]
    flags_out = ["--clean-output"]
    if alias:
        flags_in.append("--input")
        flags_out.append("--output")
    parser.add_argument(*flags_in, dest="clean_input", default=str(DEFAULT_PARSE_OUTPUT),
                       help="Directory containing text.jsonl files (default: data/processed)")
    parser.add_argument(*flags_out, dest="clean_output", default=str(DEFAULT_CLEAN_OUTPUT),
                       help="Directory for cleaned sentence corpus (default: data/clean)")
    parser.add_argument("--clean-pattern", dest="clean_pattern", default="text.jsonl",
                       help="Filename pattern to match when cleaning (default: text.jsonl)")
    parser.add_argument("--clean-min-chars", dest="clean_min_chars", type=int, default=30,
                       help="Minimum characters per sentence unless it carries numbers/durations (default: 30)")
    parser.add_argument("--clean-max-sentence-len", dest="clean_max_sentence_len", type=int, default=2000,
                       help="Maximum characters per sentence before wrapping (default: 2000)")
    parser.add_argument("--clean-hard-wrap-len", dest="clean_hard_wrap_len", type=int, default=2800,
                       help="Emergency wrap length (default: 2800)")
    parser.add_argument("--clean-heartbeat", dest="clean_heartbeat", type=int, default=1000,
                       help="Heartbeat logging interval (default: 1000 lines)")
    parser.add_argument("--clean-log-level", dest="clean_log_level", default="INFO",
                       help="Logging level for cleaning (default: INFO)")


def add_index_arguments(parser: argparse.ArgumentParser, *, alias: bool = False) -> None:
    flags_in = ["--index-input"]
    flags_out = ["--index-output"]
    if alias:
        flags_in.append("--input")
        flags_out.append("--output")
    parser.add_argument(*flags_in, dest="index_input", default=str(DEFAULT_CLEAN_OUTPUT),
                       help="Directory containing clean outputs (default: data/clean)")
    parser.add_argument(*flags_out, dest="index_output", default=str(DEFAULT_INDEX_OUTPUT),
                       help="Directory for schema outputs (default: data/silver)")
    parser.add_argument("--index-format", dest="index_format", default="jsonl",
                       choices=["jsonl", "parquet", "csv"], help="Export format (default: jsonl)")
    parser.add_argument("--index-overwrite", dest="index_overwrite", action="store_true",
                       help="Overwrite existing files in the output tree")
    parser.add_argument("--index-quiet", dest="index_quiet", action="store_true",
                       help="Reduce logging during schema conversion")


def add_chunk_arguments(parser: argparse.ArgumentParser, *, alias: bool = False) -> None:
    flags_in = ["--chunk-input"]
    flags_out = ["--chunk-output"]
    if alias:
        flags_in.append("--input")
        flags_out.append("--output")
    parser.add_argument(*flags_in, dest="chunk_input", default=str(DEFAULT_INDEX_OUTPUT),
                       help="Directory containing silver text corpus files (default: data/silver)")
    parser.add_argument(*flags_out, dest="chunk_output", default=str(DEFAULT_CHUNK_OUTPUT),
                       help="Directory for chunked outputs (default: data/chunked)")
    parser.add_argument("--chunk-workers", dest="chunk_workers", type=int, default=6,
                       help="Number of worker threads (default: 6)")
    parser.add_argument("--chunk-max-tokens", dest="chunk_max_tokens", type=int, default=900,
                       help="Maximum tokens per chunk (default: 900)")
    parser.add_argument("--chunk-max-chars", dest="chunk_max_chars", type=int, default=3600,
                       help="Maximum characters per chunk (default: 3600)")
    parser.add_argument("--chunk-overwrite", dest="chunk_overwrite", action="store_true",
                       help="Overwrite existing chunk files")


def add_embed_arguments(parser: argparse.ArgumentParser, *, alias: bool = False) -> None:
    flags_in = ["--embed-input"]
    flags_out = ["--embed-output"]
    if alias:
        flags_in.append("--input")
        flags_out.append("--output")
    parser.add_argument(*flags_in, dest="embed_input", default=str(DEFAULT_CHUNK_OUTPUT),
                       help="Directory containing text_chunks.jsonl (default: data/chunked)")
    parser.add_argument(*flags_out, dest="embed_output", default=str(DEFAULT_EMBED_OUTPUT),
                       help="Directory for FAISS index + metadata (default: data/index)")
    parser.add_argument("--embed-model", dest="embed_model", default="BAAI/bge-base-en-v1.5",
                       help="Sentence transformer model identifier")
    parser.add_argument("--embed-batch-size", dest="embed_batch_size", type=int, default=64,
                       help="Embedding batch size (default: 64)")
    parser.add_argument("--embed-use-title", dest="embed_use_title", action="store_true",
                       help="Prepend chunk titles when encoding")
    parser.add_argument("--embed-prefer-keywords", dest="embed_prefer_keywords", default="latest,final,clean",
                       help="Comma separated keywords to break rid conflicts (default: latest,final,clean)")
    parser.add_argument("--embed-strict-rid", dest="embed_strict_rid", action="store_true",
                       help="Skip chunks whose rid is not accno::text::chunk-N")


# ---------------------------------------------------------------------------
# Stage handlers
# ---------------------------------------------------------------------------

def stage_download(args: argparse.Namespace) -> None:
    from src.ingest import download_from_csv

    email = args.download_email or os.getenv("SEC_EDGAR_EMAIL") or os.getenv("SEC_EMAIL")
    if not email:
        raise SystemExit("Download stage requires an email. Use --download-email or set SEC_EMAIL env var.")
    download_from_csv.run(
        email=email,
        outdir=args.download_outdir,
        company_name=args.download_company_name,
        limit_10k=args.download_limit_10k,
        limit_10q=args.download_limit_10q,
        xbrl=not args.download_no_xbrl,
        sleep=args.download_sleep,
        companies_csv=args.download_companies_csv,
    )


def stage_postprocess(args: argparse.Namespace) -> None:
    from src.parse import postprocess_edgar

    postprocess_edgar.run(base_dir=args.postprocess_base, out_dir=args.postprocess_out)


def stage_parse(args: argparse.Namespace) -> None:
    from src.parse import text_parsing

    text_parsing.batch_parse(input_dir=args.parse_input, output_root=args.parse_output)


def stage_clean(args: argparse.Namespace) -> None:
    from src.cleaning import text_clean

    text_clean.setup_logger(args.clean_log_level)
    text_clean.clean_directory(
        input_dir=Path(args.clean_input),
        output_dir=Path(args.clean_output),
        pattern=args.clean_pattern,
        min_chars=args.clean_min_chars,
        max_sentence_len=args.clean_max_sentence_len,
        hard_wrap_len=args.clean_hard_wrap_len,
        heartbeat_every=args.clean_heartbeat,
    )


def stage_index(args: argparse.Namespace) -> None:
    from src.index import schema

    schema.process_tree(
        in_root=Path(args.index_input),
        out_root=Path(args.index_output),
        export_format=args.index_format,
        overwrite=args.index_overwrite,
        verbose=not args.index_quiet,
    )


def stage_chunk(args: argparse.Namespace) -> None:
    from src.chunking_and_embedding import chunking1

    in_root = Path(args.chunk_input)
    out_root = Path(args.chunk_output)
    files = sorted(in_root.rglob("text_corpus.jsonl"))
    if not files:
        print(f"[warn] No text_corpus.jsonl files found under {in_root}")
        return

    out_root.mkdir(parents=True, exist_ok=True)

    def _work(in_path: Path):
        out_path = chunking1.compute_out_path(in_path, in_root, out_root)
        return chunking1.chunk_one_file(
            in_path,
            out_path,
            max_tokens=args.chunk_max_tokens,
            max_chars=args.chunk_max_chars,
            overwrite=args.chunk_overwrite,
        )

    workers = max(1, args.chunk_workers)
    if workers == 1:
        for f in files:
            res = _work(f)
            print(f"[chunk] {res['status']} {res['in']} -> {res['out']}")
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_work, f): f for f in files}
            for fut in as_completed(futures):
                res = fut.result()
                print(f"[chunk] {res['status']} {res['in']} -> {res['out']}")


def stage_embed(args: argparse.Namespace) -> None:
    from src.chunking_and_embedding import embedding

    keywords = tuple(_comma_split(args.embed_prefer_keywords))
    embedding.build_index(
        input_root=args.embed_input,
        output_dir=args.embed_output,
        model_name=args.embed_model,
        batch_size=args.embed_batch_size,
        use_title=args.embed_use_title,
        prefer_keywords=keywords or ("latest", "final", "clean"),
        strict_rid=args.embed_strict_rid,
    )


STAGE_FUNCTIONS = {
    "download": stage_download,
    "postprocess": stage_postprocess,
    "parse": stage_parse,
    "clean": stage_clean,
    "index": stage_index,
    "chunk": stage_chunk,
    "embed": stage_embed,
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_pipeline(args: argparse.Namespace) -> None:
    stages = parse_stage_list(args.stages)
    for name in stages:
        print()
        print(f"[stage] {name}")
        handler = STAGE_FUNCTIONS[name]
        handler(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for the text retrieval pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run one or more stages sequentially")
    run_parser.add_argument("--stages", default=",".join(STAGE_SEQUENCE),
                            help="Comma separated list of stages (default: full pipeline)")
    add_download_arguments(run_parser)
    add_postprocess_arguments(run_parser)
    add_parse_arguments(run_parser)
    add_clean_arguments(run_parser)
    add_index_arguments(run_parser)
    add_chunk_arguments(run_parser)
    add_embed_arguments(run_parser)
    run_parser.set_defaults(entry=run_pipeline)

    download_parser = subparsers.add_parser("download", help="Download SEC filings listed in companies.csv")
    add_download_arguments(download_parser, alias=True)
    download_parser.set_defaults(entry=stage_download)

    postprocess_parser = subparsers.add_parser("postprocess", help="Normalize raw sec-edgar directories")
    add_postprocess_arguments(postprocess_parser, alias=True)
    postprocess_parser.set_defaults(entry=stage_postprocess)

    parse_parser = subparsers.add_parser("parse", help="Parse normalized filings into text.jsonl")
    add_parse_arguments(parse_parser, alias=True)
    parse_parser.set_defaults(entry=stage_parse)

    clean_parser = subparsers.add_parser("clean", help="Clean text.jsonl into sentence-level corpus")
    add_clean_arguments(clean_parser, alias=True)
    clean_parser.set_defaults(entry=stage_clean)

    index_parser = subparsers.add_parser("index", help="Convert clean outputs into schema-validated silver data")
    add_index_arguments(index_parser, alias=True)
    index_parser.set_defaults(entry=stage_index)

    chunk_parser = subparsers.add_parser("chunk", help="Chunk silver text corpora for embedding")
    add_chunk_arguments(chunk_parser, alias=True)
    chunk_parser.set_defaults(entry=stage_chunk)

    embed_parser = subparsers.add_parser("embed", help="Build vector index from chunked data")
    add_embed_arguments(embed_parser, alias=True)
    embed_parser.set_defaults(entry=stage_embed)

    return parser


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    entry = getattr(args, "entry", None)
    if not entry:
        parser.print_help()
        return
    entry(args)


if __name__ == "__main__":
    main()
