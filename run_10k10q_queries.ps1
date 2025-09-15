'''

# Usage (PowerShell):
   Set-ExecutionPolicy -Scope Process Bypass
   .\run_10k10q_queries.ps1

'''
python -m src.rag.retriever.hybrid `
  --query "What were Apple’s total revenues in its 2023 10-K?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker AAPL `
  --form "10-K" `
  --year 2023 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "In Microsoft’s 2023 10-K, which segment generated the most revenue?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker MSFT `
  --form "10-K" `
  --year 2023 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "What were Tesla’s R&D expenses in the 2023 10-K?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker TSLA `
  --form "10-K" `
  --year 2023 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "How much free cash flow did Amazon report in its 2023 10-K?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker AMZN `
  --form "10-K" `
  --year 2023 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "In Google’s 2024 10-K, what were the total advertising revenues?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker GOOGL `
  --form "10-K" `
  --year 2024 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "What risks did Meta highlight about regulatory scrutiny in its 2023 10-K?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker META `
  --form "10-K" `
  --year 2023 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "How did Netflix describe its debt obligations in the 2023 10-K?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker NFLX `
  --form "10-K" `
  --year 2023 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "What was NVIDIA’s revenue growth rate in fiscal year 2024 according to the 10-K?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker NVDA `
  --form "10-K" `
  --year 2024 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "In Tesla’s Q2 2023 10-Q, how much revenue came from automotive sales?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker TSLA `
  --form "10-Q" `
  --year 2023 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "What was Apple’s net income for fiscal year 2023 in its 10-K?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker AAPL `
  --form "10-K" `
  --year 2023 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "How did Microsoft describe risks related to supply chain in its 2023 10-K?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker MSFT `
  --form "10-K" `
  --year 2023 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "What were Amazon’s advertising revenues in 2023 according to the 10-K?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker AMZN `
  --form "10-K" `
  --year 2023 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "In Google’s 2023 10-Q Q2 filing, what was the reported operating income?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker GOOGL `
  --form "10-Q" `
  --year 2023 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "How much did Meta spend on capital expenditures in 2024 10-K?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker META `
  --form "10-K" `
  --year 2024 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "What segment revenues did NVIDIA report in its 2023 10-K?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker NVDA `
  --form "10-K" `
  --year 2023 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "How much cash and cash equivalents did Apple report in its 2024 10-K?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker AAPL `
  --form "10-K" `
  --year 2024 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "What litigation risks did Tesla disclose in its 2024 10-K?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker TSLA `
  --form "10-K" `
  --year 2024 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "In Microsoft’s Q1 2025 10-Q, what was the revenue for Office Commercial products and cloud services?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker MSFT `
  --form "10-Q" `
  --year 2025 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "What was Netflix’s net income in 2024 according to its 10-K?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker NFLX `
  --form "10-K" `
  --year 2024 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

python -m src.rag.retriever.hybrid `
  --query "How did Amazon describe risks from foreign currency exchange in its 2024 10-K?" `
  --index-dir data/index `
  --content-dir data/chunked `
  --model BAAI/bge-base-en-v1.5 `
  --rerank-model cross-encoder/ms-marco-MiniLM-L-6-v2 `
  --ticker AMZN `
  --form "10-K" `
  --year 2024 `
  --topk 8 `
  --k 60 `
  --show-chars 1000 `
  --w-bm25 1.0 `
  --w-dense 2.0 `
  --ce-weight 0.01

