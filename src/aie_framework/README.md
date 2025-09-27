# AIE框架核心模块功能介绍

## 📁 模块概览

AIE (Automated Information Extraction) 框架包含5个核心模块，实现从文档到结构化信息的完整处理流程。

---

## 📄 `segmentation.py` - 文档分段模块

**🎯 功能**：将长文档智能分割为可管理的片段

**🔧 核心类**：
- `DocumentSegmenter` - 主分段器
- `FixedLengthSegmenter` - 固定长度分段
- `SemanticSegmenter` - 语义相似度分段  
- `HybridSegmenter` - 混合分段策略

**💡 使用场景**：
- 处理长篇财务报告、研究论文
- 为后续检索和分析做预处理
- 支持不同类型内容的智能识别

**⚙️ 特性**：
- 支持CUDA加速的语义分段
- 可配置的重叠比例和分段长度
- 自动识别表格、图像、标题等内容类型

---

## 🔍 `retrieval.py` - 信息检索模块

**🎯 功能**：根据查询快速定位最相关的文档片段

**🔧 核心类**：
- `DocumentRetriever` - 主检索器
- `DenseRetriever` - 密集向量检索 (CUDA加速)
- `SparseRetriever` - 稀疏TF-IDF检索
- `HybridRetriever` - 混合检索策略
- `KeywordRetriever` - 关键词匹配检索

**💡 使用场景**：
- 在大量文档中快速找到相关信息
- 支持语义搜索和关键词搜索
- 为信息提取提供精准的上下文

**⚙️ 特性**：
- FAISS索引支持，毫秒级检索
- 多种检索策略可组合使用
- 自动相似度评分和排序

---

## 📝 `summarization.py` - 文档摘要模块

**🎯 功能**：将检索到的文档片段生成简洁准确的摘要

**🔧 核心类**：
- `DocumentSummarizer` - 主摘要器
- `RefineStrategy` - 迭代优化摘要
- `MapReduceStrategy` - 分治式摘要
- `StuffStrategy` - 直接拼接摘要

**💡 使用场景**：
- 长文档的核心信息提取
- 多文档信息融合
- 为决策提供关键信息概览

**⚙️ 特性**：
- 支持多种摘要策略
- DeepSeek API集成，高质量生成
- 可控的摘要长度和风格

---

## 📊 `extraction.py` - 信息提取模块

**🎯 功能**：从摘要文本中提取结构化数据（数值、日期、文本等）

**🔧 核心类**：
- `InformationExtractor` - 主提取器
- `LLMExtractor` - 基于大模型的智能提取
- `RegexExtractor` - 基于正则表达式的快速提取
- `HybridExtractor` - 混合提取策略

**💡 使用场景**：
- 财务数据提取（营收、利润、资产）
- 关键指标识别和量化
- 非结构化文本的结构化转换

**⚙️ 特性**：
- 支持多种数据类型（数值、文本、日期、布尔）
- 置信度评估和错误处理
- JSON格式输出，易于后续处理

---

## 🔄 `pipeline.py` - 流水线编排模块

**🎯 功能**：协调所有模块，实现端到端的文档处理流程

**🔧 核心类**：
- `AIEPipeline` - 主流水线
- `ProcessingResult` - 结果数据结构
- `BatchProcessor` - 批量处理器

**💡 使用场景**：
- 完整的文档分析工作流
- 批量处理大量文档
- 实验和生产环境的统一接口

**⚙️ 特性**：
- 模块化设计，易于扩展和定制
- 完整的错误处理和日志记录
- 支持并行处理和性能监控

---

## 🚀 典型使用流程

```python
from src.aie_framework import AIEPipeline, ExtractionTarget

# 1. 定义提取目标
targets = [
    ExtractionTarget("revenue", "2023年营业收入", "number", unit="万元"),
    ExtractionTarget("profit", "2023年净利润", "number", unit="万元")
]

# 2. 初始化流水线
pipeline = AIEPipeline(config, llm_interface)

# 3. 处理文档
result = pipeline.process_document(
    document_text="财务报告内容...",
    query="2023年财务数据",
    extraction_targets=targets
)

# 4. 获取结果
print(f"营业收入: {result.extractions[0].value}")
print(f"净利润: {result.extractions[1].value}")
```

## 🔧 配置要求

- **Python**: 3.8+
- **GPU**: NVIDIA GPU with CUDA support (推荐)
- **内存**: 8GB+ RAM
- **依赖**: PyTorch, sentence-transformers, FAISS, OpenAI

## 📈 性能特点

- **分段**: 1000字符/秒
- **检索**: 毫秒级响应 (FAISS索引)
- **摘要**: 20秒/文档 (DeepSeek API)
- **提取**: 75%+ 准确率
- **总体**: 20秒/文档 (完整流程)

---

*每个模块都支持独立使用，也可以通过pipeline进行完整的端到端处理。*
