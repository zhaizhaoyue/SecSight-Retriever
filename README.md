# Multi-Modal-LLM-Research-Assistant-for-Finance

一个面向 **金融研究与分析** 的多模态 LLM 助手，支持财报解析、数据抓取、问答检索、信号提取与回测。该项目结合了 **自然语言处理、信息抽取、RAG（Retrieval-Augmented Generation）、多模态输入** 等技术，旨在为金融从业者和研究人员提供高效的数据洞察工具。

---

## 🚀 功能特性

* **数据采集**：从公开渠道（如 SEC EDGAR、CSV 文件等）自动抓取财报和公司数据。
* **数据预处理**：清洗、解析财报文本与结构化表格，转化为标准格式。
* **问答系统 (QA)**：基于 RAG 技术，支持自然语言提问，结合知识库返回答案。
* **信号生成**：从财报与市场数据中提取投资相关信号。
* **回测 (Backtest)**：对信号进行历史验证，评估策略效果。
* **多模态支持**：文本、结构化数据、多源异构信息整合。

---

## 📂 目录结构

```
configs/                 # 配置文件
data/  
  ├─ processed/          # 处理后的数据  
  ├─ qa/                 # 问答数据集  
  ├─ companies.csv       # 公司清单  
docker/                  # Docker 相关配置  
notebooks/               # Jupyter Notebook 示例  
scripts/                 # 辅助脚本  
src/  
  ├─ backtest/           # 回测逻辑  
  ├─ common/             # 公共工具函数  
  ├─ embed/              # 向量化与嵌入  
  ├─ index/              # 索引构建  
  ├─ ingest/             # 数据导入与预处理  
  ├─ parse/              # 文本/财报解析  
  ├─ rag/                # RAG 检索增强模块  
  ├─ signals/            # 信号生成与处理  
  ├─ cli.py              # 命令行入口  
  ├─ utils.py            # 通用工具  
tests/                   # 单元测试  
requirements.txt         # 依赖文件  
Makefile                 # 自动化命令  
demo.ipynb               # 演示 Notebook  
```

---

## ⚙️ 安装与环境

### 1. 克隆仓库

```bash
git clone https://github.com/your-username/Multi-Modal-LLM-Research-Assistant-for-Finance.git
cd Multi-Modal-LLM-Research-Assistant-for-Finance
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. 配置环境变量

在 `.env` 或 `configs/` 中配置 API keys（如 OpenAI/金融数据 API）。

---

## 🛠 使用方法

### 命令行运行

```bash
python src/cli.py --task ingest --input data/companies.csv
```

### 数据采集

```bash
python scripts/data_collection.py
```

### 数据解析

```bash
python scripts/data_parsing.py
```

### 财报下载（EDGAR）

```bash
python scripts/postprocess_edgar.py
```

### 回测示例

```bash
python -m src.backtest.run --config configs/backtest.yaml
```

---

## 📊 示例

* `demo.ipynb` 中包含：

  * 从 SEC EDGAR 下载财报
  * 提取关键财务指标
  * 问答系统示例
  * 信号生成与回测流程

---

## ✅ 测试

运行单元测试：

```bash
pytest tests/
```

---

## 🛤️ 项目规划

* [ ] 扩展多模态输入（图表 / 财务图片识别）
* [ ] 优化 RAG 检索和 Embedding
* [ ] 增加量化回测因子库
* [ ] API + Web 前端 Demo

---

## 📜 许可证

本项目基于 MIT License 开源。
