# run_answer_api_batch.ps1
'''Usage:
  # 建议先在 PowerShell 里设置一次 API key
  # $env:LLM_API_KEY="sk-f6220301c405405a8ca5c65a06a75f7b"
  Set-ExecutionPolicy -Scope Process Bypass
  .\run_answer_api_batch.ps1
  '''
# 所有输出会写到 logs\answer_api_<timestamp>.log

$ErrorActionPreference = "Stop"

# 日志目录 & 文件
$ts      = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir  = Join-Path (Get-Location) "logs"
$logFile = Join-Path $logDir "answer_api_$ts.log"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
"=== Batch run of answer_api ($ts) ===" | Out-File -FilePath $logFile -Encoding utf8

# 批量问题列表
$tests = @(
  @{ Query="What were Apple’s total revenues in its 2023 10-K?";   Ticker="AAPL"; Year=2023 }
  @{ Query="What risks related to climate change did ExxonMobil highlight in its 2023 10-K?"; Ticker="XOM"; Year=2023 }
  @{ Query="What risks related to regulatory scrutiny did Meta highlight in its 2023 10-K?"; Ticker="META"; Year=2023 }
  @{ Query="Which segment generated the most revenue for Microsoft in 2024 10-K?"; Ticker="MSFT"; Year=2024 }
  #@{ Query="What were Amazon’s total revenues in its 2023 10-K?";  Ticker="AMZN"; Year=2023 }
  @{ Query="What risks did Tesla mention in its 2023 10-K?";       Ticker="TSLA"; Year=2023 }
  @{ Query="What were NVIDIA’s R&D expenses in its 2023 10-K?";   Ticker="NVDA"; Year=2023 }
  #@{ Query="What were Alphabet’s advertising revenues in its 2023 10-K?"; Ticker="GOOGL"; Year=2023 }
  @{ Query="What litigation risks did Johnson & Johnson highlight in its 2023 10-K?"; Ticker="JNJ"; Year=2023 }
  #@{ Query="What risks did JPMorgan highlight in its 2023 10-K?"; Ticker="JPM"; Year=2023 }
  @{ Query="What were Procter & Gamble’s net sales in its 2023 10-K?"; Ticker="PG"; Year=2023 }
  #@{ Query="What were Pfizer’s total revenues in its 2023 10-K?"; Ticker="PFE"; Year=2023 }
  @{ Query="What risks related to competition did Visa highlight in its 2023 10-K?"; Ticker="V"; Year=2023 }
  @{ Query="What were Walmart’s revenues in its 2023 10-K?";      Ticker="WMT"; Year=2023 }
  @{ Query="What risks related to healthcare regulation did UnitedHealth mention in its 2023 10-K?"; Ticker="UNH"; Year=2023 }
  #@{ Query="What were Berkshire Hathaway’s insurance revenues in its 2023 10-K?"; Ticker="BRK-A"; Year=2023 }
  #@{ Query="What were Coca-Cola’s net operating revenues in its 2023 10-K?"; Ticker="KO"; Year=2023 }
  @{ Query="What risks related to energy markets did Chevron highlight in its 2023 10-K?"; Ticker="CVX"; Year=2023 }
  #@{ Query="What were PepsiCo’s revenues in its 2023 10-K?";      Ticker="PEP"; Year=2023 }
  @{ Query="What risks related to consumer demand did Mastercard highlight in its 2023 10-K?"; Ticker="MA"; Year=2023 }
)

# 公共参数
$common = @(
  "-m", "src.rag.retriever.answer_api",
  "--index-dir", "data/index",
  "--content-dir", "data/chunked",
  "--model", "BAAI/bge-base-en-v1.5",
  "--rerank-model", "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "--topk", "8",
  "--bm25-topk", "200",
  "--dense-topk", "200",
  "--ce-candidates", "256",
  "--w-bm25", "1.0",
  "--w-dense", "2.0",
  "--ce-weight", "0.4",
  "--form", "10-K",
  "--llm-base-url", "https://api.deepseek.com/v1",
  "--llm-model", "deepseek-chat",
  "--llm-api-key", $env:LLM_API_KEY,
  "--json-out"
)

# 执行循环
foreach ($t in $tests) {
  $sep = "--------------------------------------------------------------------------------"
  $hdr = ">>> Running {0} ({1})" -f $t.Ticker, $t.Year

  $hdr | Tee-Object -FilePath $logFile -Append | Out-Null
  $sep | Tee-Object -FilePath $logFile -Append | Out-Null

  $args = @()
  $args += $common
  $args += @("--query", $t.Query, "--ticker", $t.Ticker, "--year", [string]$t.Year)

  & python $args 2>&1 | Tee-Object -FilePath $logFile -Append | Out-Null
  "`n" | Out-File -FilePath $logFile -Append
}

"=== All done ===" | Out-File -FilePath $logFile -Append
Write-Host "Logs saved to: $logFile" -ForegroundColor Green
