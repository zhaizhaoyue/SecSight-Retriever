# Multi-Modal LLM Research Assistant for Finance

一个面向金融研究的多模态大语言模型助手项目。
本项目聚焦 **美股财报（10-K / 10-Q）自动化下载 → 标准化解析 → 质检 → 数据服务**，为后续因子研究与 RAG 问答提供高质量的数据基础。

---

## 📂 目录结构

```
data/
  raw_reports/         # 原始下载的财报 (sec-edgar-filings)
  processed/           # 解析后的 JSONL / Parquet
  qa/                  # 质检报告与日志
src/
  data_collection.py   # 下载脚本
  download_from_csv.py # 批量下载 (基于 companies.csv)
  data_parsing.py      # HTML/文本解析
  qa_checks.py         # 质检脚本
  postprocess_edgar.py # 标准化 & 重命名
companies.csv          # 配置文件：公司/年份/表单类型
README.md
LICENSE
```

---

## 🚀 功能

* 从 SEC EDGAR 自动下载指定公司财报
* 支持 **批量驱动 (companies.csv)** 管理 Ticker/Years/Form
* 统一重命名，生成标准化文件路径
* 文本解析 → JSONL；表格解析 → Parquet
* 自动质检（文本行数 / 表格数量 / 数值比例）
* 日志记录下载成功/失败

---

## 🔧 使用方法

1. **配置下载公司**
   编辑 `data/companies.csv`，示例：

   ```csv
   ticker,market,source,form_types,years
   AAPL,US,EDGAR,"10-K|10-Q","2023|2024|2025"
   MSFT,US,EDGAR,"10-K|10-Q","2023|2024|2025"
   ```

2. **运行批量下载**

   ```bash
   python src/download_from_csv.py --email "your@email.com"
   ```

3. **后处理（标准化）**

   ```bash
   python src/postprocess_edgar.py
   ```

4. **解析文本/表格**

   ```bash
   python src/data_parsing.py
   ```

5. **质检**

   ```bash
   python src/qa_checks.py
   ```

---

## 📊 依赖

* Python 3.10+
* [sec-edgar-downloader](https://pypi.org/project/sec-edgar-downloader/)
* pandas
* pyarrow
* beautifulsoup4
* lxml

安装：

```bash
pip install -r requirements.txt
```

---

## 📜 License

本项目采用 [MIT License](LICENSE) 开源。
