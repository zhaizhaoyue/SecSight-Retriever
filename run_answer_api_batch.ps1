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
$tests = @(
@{ Query="What risks did Apple highlight related to supply chain disruptions in 2021?"; Ticker="AAPL"; Year=2021 }
@{ Query="How did Microsoft describe the competitive landscape for Azure and cloud services in 2023?"; Ticker="MSFT"; Year=2023 }
@{ Query="What strategic priorities did Alphabet outline for its advertising business in 2022?"; Ticker="GOOGL"; Year=2022 }
@{ Query="What challenges did Amazon report in managing international expansion in 2023?"; Ticker="AMZN"; Year=2023 }
@{ Query="What risks did NVIDIA identify regarding data center supply chain or customer concentration in 2024?"; Ticker="NVDA"; Year=2024 }
@{ Query="How did Meta describe trends in user engagement and monetization in 2023?"; Ticker="META"; Year=2023 }
@{ Query="What regulatory or market risks did Berkshire Hathaway discuss in 2021?"; Ticker="BRK-A"; Year=2021 }
@{ Query="What risks did Tesla highlight regarding scaling vehicle production globally in 2022?"; Ticker="TSLA"; Year=2022 }
@{ Query="What economic or credit risks did JPMorgan emphasize in its 2023 10-K?"; Ticker="JPM"; Year=2023 }
@{ Query="How did Johnson & Johnson describe its pharmaceutical pipeline strategy in 2024?"; Ticker="JNJ"; Year=2024 }
@{ Query="What risks did Visa identify related to global payment volumes and consumer demand in 2020?"; Ticker="V"; Year=2020 }
@{ Query="What competitive or regulatory risks did Mastercard emphasize in 2022?"; Ticker="MA"; Year=2022 }
@{ Query="How did Procter & Gamble describe its pricing and product innovation strategies in 2023?"; Ticker="PG"; Year=2023 }
@{ Query="What risks or opportunities did Coca-Cola outline in relation to emerging markets in 2022?"; Ticker="KO"; Year=2022 }
@{ Query="How did PepsiCo describe consumer trends influencing its product portfolio in 2021?"; Ticker="PEP"; Year=2021 }
@{ Query="What climate change or environmental risks did ExxonMobil highlight in 2020?"; Ticker="XOM"; Year=2020 }
@{ Query="How did Chevron describe geopolitical or regulatory risks affecting its operations in 2023?"; Ticker="CVX"; Year=2023 }
@{ Query="What risks did Walmart highlight related to labor, wages, or workforce management in 2022?"; Ticker="WMT"; Year=2022 }
@{ Query="What legal or regulatory risks did Pfizer outline regarding its COVID-19 products in 2021?"; Ticker="PFE"; Year=2021 }
@{ Query="What risks did UnitedHealth describe related to healthcare regulation in 2024?"; Ticker="UNH"; Year=2024 }
@{ Query="How did Apple describe its R&D priorities and innovation strategy in 2021?"; Ticker="AAPL"; Year=2021 }
@{ Query="What growth drivers did Microsoft highlight for server products and cloud services in 2023?"; Ticker="MSFT"; Year=2023 }
@{ Query="What risks did Alphabet outline regarding rising traffic acquisition costs in 2022?"; Ticker="GOOGL"; Year=2022 }
@{ Query="What challenges did Amazon discuss regarding profitability in its international operations in 2023?"; Ticker="AMZN"; Year=2023 }
@{ Query="What opportunities and risks did NVIDIA discuss for gaming and visualization markets in 2024?"; Ticker="NVDA"; Year=2024 }
@{ Query="How did Meta describe its capital expenditure strategy and infrastructure priorities in 2023?"; Ticker="META"; Year=2023 }
@{ Query="What risks did Berkshire Hathaway highlight related to its insurance operations in 2021?"; Ticker="BRK-A"; Year=2021 }
@{ Query="What challenges did Tesla describe in expanding its energy generation and storage business in 2022?"; Ticker="TSLA"; Year=2022 }
@{ Query="How did JPMorgan describe trends in investment banking and asset management in 2023?"; Ticker="JPM"; Year=2023 }
@{ Query="What strategic priorities did Johnson & Johnson outline for its medical devices segment in 2024?"; Ticker="JNJ"; Year=2024 }
@{ Query="What risks did Visa highlight regarding client incentives and competitive dynamics in 2020?"; Ticker="V"; Year=2020 }
@{ Query="How did Mastercard describe its approach to rebates and incentives in 2022?"; Ticker="MA"; Year=2022 }
@{ Query="What risks and opportunities did Procter & Gamble identify regarding commodity costs in 2023?"; Ticker="PG"; Year=2023 }
@{ Query="How did Coca-Cola describe its pricing strategy and response to market conditions in 2022?"; Ticker="KO"; Year=2022 }
@{ Query="What risks did PepsiCo identify related to supply chain disruptions in 2021?"; Ticker="PEP"; Year=2021 }
@{ Query="How did ExxonMobil describe operational challenges in its downstream business in 2020?"; Ticker="XOM"; Year=2020 }
@{ Query="What were Chevron's stated capital allocation priorities in 2023?"; Ticker="CVX"; Year=2023 }
@{ Query="What supply chain or logistics risks did Walmart identify in 2022?"; Ticker="WMT"; Year=2022 }
@{ Query="What risks did Pfizer mention regarding future drug development and regulatory approvals in 2021?"; Ticker="PFE"; Year=2021 }
@{ Query="How did UnitedHealth describe growth opportunities in its Optum segment in 2024?"; Ticker="UNH"; Year=2024 }
@{ Query="How did Apple describe trends in its services ecosystem in 2021?"; Ticker="AAPL"; Year=2021 }
@{ Query="What did Microsoft emphasize about its commercial cloud strategy in 2023?"; Ticker="MSFT"; Year=2023 }
@{ Query="What risks did Alphabet mention regarding cloud profitability and competition in 2022?"; Ticker="GOOGL"; Year=2022 }
@{ Query="How did Amazon describe the role of advertising in its overall strategy in 2023?"; Ticker="AMZN"; Year=2023 }
@{ Query="What opportunities did NVIDIA highlight for its automotive business in 2024?"; Ticker="NVDA"; Year=2024 }
@{ Query="What risks and opportunities did Meta discuss regarding Reality Labs in 2023?"; Ticker="META"; Year=2023 }
@{ Query="What risks did Berkshire Hathaway mention related to its railroad and energy operations in 2021?"; Ticker="BRK-A"; Year=2021 }
@{ Query="What challenges did Tesla outline in scaling solar and storage deployments in 2022?"; Ticker="TSLA"; Year=2022 }
@{ Query="What risks did JPMorgan highlight related to credit quality and provisions in 2023?"; Ticker="JPM"; Year=2023 }
@{ Query="What financial and strategic impacts did Johnson & Johnson discuss regarding its consumer health spin-off in 2024?"; Ticker="JNJ"; Year=2024 }
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
