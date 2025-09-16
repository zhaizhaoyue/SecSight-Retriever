# run_answer_api_batch.ps1
'''Usage:
  # 建议先在 PowerShell 里设置一次 API key
  $env:LLM_API_KEY="sk-f6220301c405405a8ca5c65a06a75f7b"
  Set-ExecutionPolicy -Scope Process Bypass
  .\run_answer_api_batch.ps1
  '''
# 所有输出会写到 logs\answer_api_<timestamp>.log
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$ErrorActionPreference = "Stop"

# 日志目录 & 文件
$ts      = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir  = Join-Path (Get-Location) "logs"
$logFile = Join-Path $logDir "answer_api_$ts.log"
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
"=== Batch run of answer_api ($ts) ===" | Out-File -FilePath $logFile -Encoding utf8

# 批量问题列表
$tests = @(
  @{ Query="What were Apple’s total revenues in its 2023 10-K?";   Ticker="AAPL"; Year=2024 }
  @{ Query="What risks related to climate change did ExxonMobil highlight in its 2023 10-K?"; Ticker="XOM"; Year=2023 }
  @{ Query="What risks related to regulatory scrutiny did Meta highlight in its 2023 10-K?"; Ticker="META"; Year=2023 }
  @{ Query="Which segment generated the most revenue for Microsoft in 2024 10-K?"; Ticker="MSFT"; Year=2024 }
  @{ Query="What were Amazon’s total revenues in its 2023 10-K?";  Ticker="AMZN"; Year=2024 }
  @{ Query="What risks did Tesla mention in its 2023 10-K?";       Ticker="TSLA"; Year=2023 }
  @{ Query="What were NVIDIA’s R&D expenses in its 2023 10-K?";   Ticker="NVDA"; Year=2023 }
  @{ Query="What were Alphabet’s advertising revenues in its 2023 10-K?"; Ticker="GOOGL"; Year=2023 }
  @{ Query="What litigation risks did Johnson & Johnson highlight in its 2023 10-K?"; Ticker="JNJ"; Year=2023 }
  @{ Query="What risks did JPMorgan highlight in its 2023 10-K?"; Ticker="JPM"; Year=2023 }
  @{ Query="What were Procter & Gamble’s net sales in its 2023 10-K?"; Ticker="PG"; Year=2023 }
  @{ Query="What were Pfizer’s total revenues in its 2023 10-K?"; Ticker="PFE"; Year=2024 }
  @{ Query="What risks related to competition did Visa highlight in its 2023 10-K?"; Ticker="V"; Year=2023 }
  @{ Query="What were Walmart’s revenues in its 2023 10-K?";      Ticker="WMT"; Year=2023 }
  @{ Query="What risks related to healthcare regulation did UnitedHealth mention in its 2023 10-K?"; Ticker="UNH"; Year=2023 }
  @{ Query="What were Berkshire Hathaway’s insurance revenues in its 2023 10-K?"; Ticker="BRK-A"; Year=2023 }
  @{ Query="What were Coca-Cola’s net operating revenues in its 2023 10-K?"; Ticker="KO"; Year=2023 }
  @{ Query="What risks related to energy markets did Chevron highlight in its 2023 10-K?"; Ticker="CVX"; Year=2023 }
  @{ Query="What were PepsiCo’s revenues in its 2023 10-K?";      Ticker="PEP"; Year=2023 }
  @{ Query="What risks related to consumer demand did Mastercard highlight in its 2023 10-K?"; Ticker="MA"; Year=2023 }
  @{ Query="What were Intel’s R&D expenses in its 2023 10-K?"; Ticker="INTC"; Year=2023 }
  @{ Query="What risks related to supply chain did Apple highlight in its 2023 10-K?"; Ticker="AAPL"; Year=2023 }
  @{ Query="What were Cisco’s total revenues in its 2023 10-K?"; Ticker="CSCO"; Year=2023 }
  @{ Query="What risks related to cybersecurity did IBM mention in its 2023 10-K?"; Ticker="IBM"; Year=2023 }
  @{ Query="What were Netflix’s content expenses in its 2023 10-K?"; Ticker="NFLX"; Year=2023 }
  @{ Query="What risks related to regulation did PayPal mention in its 2023 10-K?"; Ticker="PYPL"; Year=2023 }
  @{ Query="What were Oracle’s cloud revenues in its 2023 10-K?"; Ticker="ORCL"; Year=2023 }
  @{ Query="What risks related to competition did Adobe highlight in its 2023 10-K?"; Ticker="ADBE"; Year=2023 }
  @{ Query="What were Qualcomm’s licensing revenues in its 2023 10-K?"; Ticker="QCOM"; Year=2023 }
  @{ Query="What risks related to labor shortages did Starbucks highlight in its 2023 10-K?"; Ticker="SBUX"; Year=2023 }
  @{ Query="What were Boeing’s total revenues in its 2023 10-K?"; Ticker="BA"; Year=2023 }
  @{ Query="What risks related to raw material costs did General Motors highlight in its 2023 10-K?"; Ticker="GM"; Year=2023 }
  @{ Query="What were Ford’s automotive revenues in its 2023 10-K?"; Ticker="F"; Year=2023 }
  @{ Query="What risks related to intellectual property did Moderna highlight in its 2023 10-K?"; Ticker="MRNA"; Year=2023 }
  @{ Query="What were Eli Lilly’s total revenues in its 2023 10-K?"; Ticker="LLY"; Year=2023 }
  @{ Query="What risks related to data privacy did Salesforce highlight in its 2023 10-K?"; Ticker="CRM"; Year=2023 }
  @{ Query="What were American Express’s total revenues in its 2023 10-K?"; Ticker="AXP"; Year=2023 }
  @{ Query="What risks related to geopolitical tensions did Caterpillar mention in its 2023 10-K?"; Ticker="CAT"; Year=2023 }
  @{ Query="What were Goldman Sachs’s total revenues in its 2023 10-K?"; Ticker="GS"; Year=2023 }
  @{ Query="What risks related to interest rates did Bank of America highlight in its 2023 10-K?"; Ticker="BAC"; Year=2023 }
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

  $hdr | Tee-Object -FilePath $logFile -Append -Encoding utf8 | Out-Null
  $sep | Tee-Object -FilePath $logFile -Append -Encoding utf8 | Out-Null

  $args = @()
  $args += $common
  $args += @("--query", $t.Query, "--ticker", $t.Ticker, "--year", [string]$t.Year)

  & python -X utf8 $args 2>&1 | Tee-Object -FilePath $logFile -Append -Encoding utf8 | Out-Null

  "`n" | Out-File -FilePath $logFile -Append -Encoding utf8
}

"=== All done ===" | Out-File -FilePath $logFile -Append -Encoding utf8
Write-Host "Logs saved to: $logFile" -ForegroundColor Green
