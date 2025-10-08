# Multi-Modal-LLM-Research-Assistant-for-Finance

This repository powers a financial text-retrieval pipeline that turns raw SEC filings into a searchable vector index. The tooling focuses on repeatable ingestion, parsing, cleaning, chunking, and embedding of US corporate reports so they can be consumed by an LLM-based research assistant.

## Pipeline Overview

| Stage | Script | Input -> Output | Purpose |
| --- | --- | --- | --- |
| 1. download | src/ingest/download_from_csv.py | data/companies.csv -> data/raw_reports/sec-edgar-filings/ | Pull the latest 10-K/10-Q filings from EDGAR. |
| 2. postprocess | src/parse/postprocess_edgar.py | data/raw_reports/sec-edgar-filings/ -> data/raw_reports/standard/ | Normalize folder names and collect the primary HTML/XBRL payloads. |
| 3. parse | src/parse/text_parsing.py | data/raw_reports/standard/ -> data/processed/ | Extract structured metadata and generate 	ext.jsonl chunks. |
| 4. clean | src/cleaning/text_clean.py | data/processed/ -> data/clean/ | Sentence-level cleaning with numeric extraction helpers. |
| 5. index | src/index/schema.py | data/clean/ -> data/silver/ | Validate against Pydantic schemas and emit clean “silver” records. |
| 6. chunk | src/chunking_and_embedding/chunking1.py | data/silver/ -> data/chunked/ | Build retrieval-ready text chunks with heading context. |
| 7. mbed | src/chunking_and_embedding/embedding.py | data/chunked/ -> data/index/ | Encode chunks, build the FAISS index, and write metadata/id maps. |

All seven stages can be orchestrated from the new src/cli.py interface.

## Quick Start

1. **Install dependencies**
   `ash
   python -m venv .venv
   .venv\Scripts\activate  # PowerShell on Windows
   pip install -r requirements.txt
   `

2. **Provide EDGAR credentials**
   `powershell
   ="your-email@example.com"
   `
   The same value can also be supplied with --download-email when running the CLI.

3. **Run the full pipeline**
   `ash
   python src/cli.py run --stages download,postprocess,parse,clean,index,chunk,embed      --download-email your-email@example.com
   `

The command uses the default directory layout under data/ and will stream logs for each stage.

## CLI Highlights

### Inspect available commands
`ash
python src/cli.py --help
`

### Run a single stage
`ash
# Only chunk existing silver outputs
python src/cli.py chunk --input data/silver --output data/chunked --chunk-workers 4

# Rebuild embeddings with a different model
python src/cli.py embed --embed-model BAAI/bge-small-en-v1.5 --embed-use-title
`

### Customize a pipeline run
You can pass any stage option to 
un. For example, to fetch only the latest 10-K and skip XBRL downloads:
`ash
python src/cli.py run   --download-email your-email@example.com   --download-limit-10k 1   --download-no-xbrl   --stages download,postprocess,parse
`

## Data Flow

`
data/companies.csv
  -> data/raw_reports/sec-edgar-filings/
  -> data/raw_reports/standard/
  -> data/processed/
  -> data/clean/
  -> data/silver/
  -> data/chunked/
  -> data/index/
`

Each directory is safe to cache between runs; the CLI only overwrites data when the relevant --*-overwrite flag is provided.

## Testing the Stages

Every module exposes a callable entry point that mirrors the CLI:

- download_from_csv.run(...)
- postprocess_edgar.run(...)
- 	ext_parsing.batch_parse(...)
- 	ext_clean.clean_directory(...)
- schema.process_tree(...)
- chunking1.chunk_one_file(...)
- mbedding.build_index(...)

This makes it straightforward to write unit tests around individual transformations or to embed the pipeline inside a larger orchestration framework.

## Notes

- The ingest step respects EDGAR rate limits; adjust --download-sleep to stay compliant with SEC guidance.
- The embedding stage loads FAISS and sentence-transformers lazily to keep imports fast when you only need upstream stages.
- Directory defaults assume the repository root as the working directory; override paths via CLI flags when running in other environments.

With the pipeline in place you can quickly curate new filings, refresh embeddings, and serve them to downstream retrieval or question-answering components.
