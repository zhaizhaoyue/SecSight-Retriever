"""
Document Segmentation Module - 文档分段模块

🎯 功能：将长文档智能分割为可管理的片段
📝 支持：固定长度、语义相似度、混合分段等多种策略
🚀 特性：CUDA加速的语义分段，自动内容类型识别
💡 用途：长篇文档预处理，为检索和分析提供基础

核心类：
- DocumentSegmenter: 主分段器，根据配置选择分段策略
- FixedLengthSegmenter: 固定长度分段，适合快速处理
- SemanticSegmenter: 基于语义相似度的智能分段 (CUDA加速)
- HybridSegmenter: 混合分段，支持表格、图像、标题识别

典型用法：
# 1) 默认递归 data/**/train.jsonl，输出到 data/segmented_data 镜像
python segmentation.py

# 2) 指定分段策略与长度（hybrid + 900 近似 token，10% 重叠）
python segmentation.py --split-method hybrid --max-segment-length 900 --overlap-ratio 0.1

# 3) 开启 semantic 断点（需要 sentence-transformers）
python segmentation.py --split-method hybrid --semantic-breaks --semantic-device cuda

# 4) 自定义文本字段候选（按顺序匹配）
python segmentation.py --text-keys "document,text,content,report"

# 5) 输出中保留原始输入记录（便于对齐/调试）
python segmentation.py --keep-input
# 6) 追求精准度:
# 使用 Hybrid 分段器 + 语义断点（需要 GPU 或 CPU 支持 sentence-transformers）
python -m src.aie_framework.segmentation `
  --split-method hybrid `
  --max-segment-length 800 `
  --overlap-ratio 0.05 `
  --data-root "data/testing_data" `
  --out-root "data/segmented_data" `
  --glob-name "train.jsonl" `

"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import os, json, argparse
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class DocumentSegment:
    """Document segment data class"""
    id: str
    content: str
    segment_type: str  # text, table, image, header
    start_pos: int
    end_pos: int
    metadata: Dict[str, Any]
    
    def __len__(self) -> int:
        return len(self.content)
    
    def __str__(self) -> str:
        return f"Segment({self.id}, {self.segment_type}, {len(self.content)} chars)"


class BaseSegmenter(ABC):
    """Base segmenter class"""
    
    @abstractmethod
    def segment(self, document: str, metadata: Optional[Dict] = None) -> List[DocumentSegment]:
        """Segmentation method"""
        pass


class FixedLengthSegmenter(BaseSegmenter):
    """固定length分段器"""
    
    def __init__(self, max_length: int = 1000, overlap_ratio: float = 0.1):
        self.max_length = max_length
        self.overlap_size = int(max_length * overlap_ratio)
        
    def segment(self, document: str, metadata: Optional[Dict] = None) -> List[DocumentSegment]:
        """Segment by fixed length"""
        segments = []
        start = 0
        segment_id = 0
        
        while start < len(document):
            end = min(start + self.max_length, len(document))
            
            # 尝试在合适的位置分割（句号、换行等）
            if end < len(document):
                # 向后查找合适的分割点
                for i in range(end, max(start, end - 100), -1):
                    if document[i] in '.!?\n':
                        end = i + 1
                        break
            
            content = document[start:end].strip()
            if content:
                segment = DocumentSegment(
                    id=f"seg_{segment_id:04d}",
                    content=content,
                    segment_type="text",
                    start_pos=start,
                    end_pos=end,
                    metadata=metadata or {}
                )
                segments.append(segment)
                segment_id += 1
            
            # 计算下一个开始位置（考虑重叠）
            start = max(start + 1, end - self.overlap_size)
            
        return segments


class SemanticSegmenter(BaseSegmenter):
    """语义分段器，基于句子嵌入的相似度"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 max_length: int = 1000, similarity_threshold: float = 0.7, device: str = "cuda"):
        self.max_length = max_length
        self.similarity_threshold = similarity_threshold
        self.device = device
        self.sentence_model = SentenceTransformer(model_name, device=device)
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """分割文本为句子"""
        # 简单的句子分割（可以使用更复杂的方法如spaCy）
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _calculate_similarity(self, sent1: str, sent2: str) -> float:
        """计算两个句子的相似度"""
        embeddings = self.sentence_model.encode([sent1, sent2])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return similarity
    
    def segment(self, document: str, metadata: Optional[Dict] = None) -> List[DocumentSegment]:
        """Segment based on semantic similarity"""
        sentences = self._split_into_sentences(document)
        if not sentences:
            return []
        
        segments = []
        current_segment = [sentences[0]]
        segment_id = 0
        current_length = len(sentences[0])
        
        for i in range(1, len(sentences)):
            sentence = sentences[i]
            
            # 检查是否需要开始新段落
            should_start_new = False
            
            # length检查
            if current_length + len(sentence) > self.max_length:
                should_start_new = True
            else:
                # 语义相似度检查
                last_sentence = current_segment[-1]
                similarity = self._calculate_similarity(last_sentence, sentence)
                if similarity < self.similarity_threshold:
                    should_start_new = True
            
            if should_start_new and current_segment:
                # 创建当前段落
                segment_text = ' '.join(current_segment)
                start_pos = document.find(current_segment[0])
                end_pos = start_pos + len(segment_text)
                
                segment = DocumentSegment(
                    id=f"sem_seg_{segment_id:04d}",
                    content=segment_text,
                    segment_type="text",
                    start_pos=start_pos,
                    end_pos=end_pos,
                    metadata=metadata or {}
                )
                segments.append(segment)
                
                # 开始新段落
                current_segment = [sentence]
                current_length = len(sentence)
                segment_id += 1
            else:
                current_segment.append(sentence)
                current_length += len(sentence)
        
        # processing最后一个段落
        if current_segment:
            segment_text = ' '.join(current_segment)
            start_pos = document.find(current_segment[0])
            end_pos = start_pos + len(segment_text)
            
            segment = DocumentSegment(
                id=f"sem_seg_{segment_id:04d}",
                content=segment_text,
                segment_type="text",
                start_pos=start_pos,
                end_pos=end_pos,
                metadata=metadata or {}
            )
            segments.append(segment)
        
        return segments


class HybridSegmenter(BaseSegmenter):
    """
    AIE-style hybrid segmenter with:
      1) Serialization (PLAIN for tables)
      2) Split   (long elements -> chunks)
      3) Merge   (adjacent short elements -> concat)
    Also tracks markdown headers and injects section_path into metadata.
    """

    def __init__(
        self,
        max_length: int = 1000,              # 近似 token 上限（字符/词近似）
        overlap_ratio: float = 0.1,          # 文本切片重叠
        min_merge_tokens: int = 180,         # 过短段合并阈值
        max_table_rows_per_chunk: int = 30,  # 表格行块大小（含表头）
        add_section_headers: bool = True,    # 将上游标题注入 metadata
        semantic_breaks: bool = False,       # 对文本切片启用语义断点
        semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        semantic_device: str = "cuda",
    ):
        self.max_length = max_length
        self.overlap_ratio = overlap_ratio
        self.overlap_size = max(0, int(max_length * overlap_ratio))
        self.min_merge_tokens = min_merge_tokens
        self.max_table_rows_per_chunk = max_table_rows_per_chunk
        self.add_section_headers = add_section_headers
        self.semantic_breaks = semantic_breaks

        self._sentence_model = None
        if semantic_breaks:
            try:
                self._sentence_model = SentenceTransformer(semantic_model, device=semantic_device)
            except Exception:
                logger.warning("SentenceTransformer init failed; semantic_breaks disabled.")
                self.semantic_breaks = False

    # ---------- low-level utils ----------
    _sent_pat = re.compile(r'(?<=[。．.!?？!])\s+|[\r\n]+')
    _header_pat = re.compile(r'^(#{1,6})\s+(.*)$')
    _html_table_pat = re.compile(r'<table.*?>.*?</table>', re.S | re.I)

    @staticmethod
    def _token_len(text: str) -> int:
        # 简单近似：词+标点。需要更准可替换为 tiktoken。
        return len(re.findall(r"\w+|[^\s\w]", text))

    def _split_text_by_tokens(self, text: str) -> List[str]:
        if self._token_len(text) <= self.max_length:
            return [text.strip()]

        # 先句切，再按近似 token 控制窗口，必要时添加重叠
        sents = [s.strip() for s in re.split(self._sent_pat, text) if s.strip()]
        chunks, cur, cur_len = [], [], 0

        def flush():
            if not cur: return
            chunk = " ".join(cur).strip()
            if chunk:
                chunks.append(chunk)

        for i, s in enumerate(sents):
            s_len = self._token_len(s)
            need_new = (cur_len + s_len > self.max_length)

            if self.semantic_breaks and cur:
                # 语义断点：若与上一句相似度过低也换段
                try:
                    emb = self._sentence_model.encode([cur[-1], s])
                    sim = float(np.dot(emb[0], emb[1]) /
                                (np.linalg.norm(emb[0]) * np.linalg.norm(emb[1]) + 1e-8))
                    if sim < 0.55:  # 经验阈值，可入参
                        need_new = True
                except Exception:
                    pass

            if need_new:
                flush()
                # 构造重叠：把上段末尾若干 token 作为开头
                if self.overlap_size > 0 and chunks:
                    tail = cur[-1] if cur else ""
                    tail_tokens = tail.split()
                    overlap = " ".join(tail_tokens[-min(len(tail_tokens), self.overlap_size):])
                    cur, cur_len = ([overlap] if overlap else []), self._token_len(overlap)
                else:
                    cur, cur_len = [], 0

            cur.append(s)
            cur_len += s_len

        flush()
        return chunks

    @staticmethod
    def _parse_md_table(block: str) -> Optional[Dict[str, Any]]:
        """Parse a markdown table block to rows/cols; return None if not a table."""
        lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
        if len(lines) < 2 or not any("|" in ln for ln in lines[:2]):
            return None

        # 去除首尾竖线，分列
        rows = []
        for ln in lines:
            if set(ln) <= {"-", "|", ":", " "}:
                # 对齐分隔行，跳过
                continue
            cells = [c.strip() for c in ln.strip("|").split("|")]
            rows.append(cells)

        if not rows:
            return None

        header = rows[0]
        data = rows[1:] if len(rows) > 1 else []
        # 对齐列数
        width = max(len(r) for r in [header] + data)
        header += ["col_%d" % i for i in range(len(header), width)]
        data = [r + [""] * (width - len(r)) for r in data]

        return {"header": header, "rows": data}

    @staticmethod
    def _serialize_plain_table(tbl: Dict[str, Any], title: Optional[str] = None) -> List[str]:
        """PLAIN: each row as 'col1=val1; col2=val2; ...' (optionally with row label)."""
        header = tbl["header"]
        lines = []
        if title:
            lines.append(f"[Table] {title}")

        for ridx, row in enumerate(tbl["rows"]):
            kvs = [f"{h}={v}" for h, v in zip(header, row)]
            line = "; ".join(kvs)
            lines.append(line)
        return lines

    def _detect_content_type(self, line: str) -> str:
        if "|" in line and not line.lstrip().startswith(("#", ">")):
            return "table"
        if self._html_table_pat.search(line):
            return "table"
        m = self._header_pat.match(line)
        if m:
            return "header"
        if re.search(r'\[图\d+\]|\[图像\]|<img', line, re.I):
            return "image"
        return "text"

    def _split_by_structure(self, document: str) -> List[Tuple[str, str, Dict[str, Any], int, int]]:
        lines = document.splitlines()
        sections = []
        buf, cur_type = [], "text"
        pos = 0
        idx = 0

        section_stack: List[Tuple[int, str]] = []  # (level, title)

        def current_section_path() -> List[str]:
            return [t for _, t in section_stack]

        def flush():
            nonlocal buf, cur_type, idx, pos
            if not buf:
                return
            raw_block = "\n".join(buf)                # 用 raw_block 算 end
            content = raw_block.strip()
            if content:
                meta = {"section_path": current_section_path()} if self.add_section_headers else {}
                start = document.find(buf[0], pos)
                end = start + len(raw_block)          # ⚠️ end 用未 strip 的长度
                sections.append((content, cur_type, meta, start, end))
                idx += 1
                pos = end + 1
            buf = []

        i = 0
        while i < len(lines):
            ln = lines[i]
            m = self._header_pat.match(ln)
            if m:
                flush()
                level = len(m.group(1)); title = m.group(2).strip()
                while section_stack and section_stack[-1][0] >= level:
                    section_stack.pop()
                section_stack.append((level, title))
                start = document.find(ln, pos)
                end = start + len(ln)
                sections.append((title, "header",
                                {"level": level, "section_path": [t for _, t in section_stack]},
                                start, end))
                pos = end + 1
                i += 1
                continue

            if self._detect_content_type(ln) == "table":
                flush()
                start = document.find(ln, pos)
                tbl_lines = [ln]
                j = i + 1
                while j < len(lines) and self._detect_content_type(lines[j]) == "table":
                    tbl_lines.append(lines[j]); j += 1
                raw_block = "\n".join(tbl_lines)      # 同理：先 raw，再 strip 存内容
                content = raw_block.strip()
                end = start + len(raw_block)
                sections.append((content, "table",
                                {"section_path": [t for _, t in section_stack]},
                                start, end))
                pos = end + 1
                i = j
                continue

            # 普通文本
            if not buf: cur_type = "text"
            buf.append(ln); i += 1

        flush()
        return sections


    # ---------- main segment ----------
    def segment(self, document: str, metadata: Optional[Dict] = None) -> List[DocumentSegment]:
        elements = self._split_by_structure(document)
        segs: List[DocumentSegment] = []
        seg_id = 0

        def add_segment(content: str, seg_type: str, start: int, end: int, extra_meta: Dict[str, Any]):
            nonlocal seg_id
            md = dict(metadata or {})
            md.update(extra_meta or {})
            segs.append(DocumentSegment(
                id=f"hyb_seg_{seg_id:04d}",
                content=content.strip(),
                segment_type=seg_type,
                start_pos=start,
                end_pos=end,
                metadata=md
            ))
            seg_id += 1

        for content, typ, meta, start, end in elements:
            if typ == "header":
                add_segment(content, "header", start, end, meta)
                continue

            if typ == "table":
                # Try to parse markdown table
                tbl = self._parse_md_table(content)
                if tbl:
                    title = meta.get("table_title") or (meta.get("section_path") or [])[-1] if meta.get("section_path") else None
                    lines = self._serialize_plain_table(tbl, title=title)
                    # chunk by rows to fit max_length
                    chunk, chunk_rows = [], []
                    cur_tokens = 0
                    # ensure first line could be a [Table] title
                    for ln in lines:
                        ln_tokens = self._token_len(ln)
                        new_len = (cur_tokens + ln_tokens)
                        if (chunk and new_len > self.max_length) or (len(chunk_rows) >= self.max_table_rows_per_chunk):
                            add_segment("\n".join(chunk), "table", start, end,
                                        {**meta, "serialization": "PLAIN",
                                         "row_range": (chunk_rows[0], chunk_rows[-1]) if chunk_rows else None})
                            chunk, chunk_rows, cur_tokens = [], [], 0
                        chunk.append(ln)
                        # row index: exclude the optional [Table] title
                        if not ln.startswith("[Table]"):
                            chunk_rows.append(len(chunk_rows))
                        cur_tokens = new_len
                    if chunk:
                        add_segment("\n".join(chunk), "table", start, end,
                                    {**meta, "serialization": "PLAIN",
                                     "row_range": (chunk_rows[0], chunk_rows[-1]) if chunk_rows else None})
                else:
                    # Fallback: treat as text
                    for sub in self._split_text_by_tokens(content):
                        add_segment(sub, "text", start, end, meta)
                continue

            # typ == "text"
            if self._token_len(content) <= self.max_length:
                add_segment(content, "text", start, end, meta)
            else:
                for sub in self._split_text_by_tokens(content):
                    add_segment(sub, "text", start, end, meta)

        # ---- Merge adjacent shorts (text + text / header + text) ----
        if not segs:
            return segs

        merged: List[DocumentSegment] = []
        buf = segs[0]
        for nxt in segs[1:]:
            can_merge = (
                (buf.segment_type == "text" and nxt.segment_type == "text") or
                (buf.segment_type == "header" and nxt.segment_type == "text")
            )
            if can_merge and (self._token_len(buf.content) + self._token_len(nxt.content) <= max(self.min_merge_tokens, self.max_length)):
                # merge
                new_content = (buf.content + "\n" + nxt.content).strip()
                buf = DocumentSegment(
                    id=buf.id,
                    content=new_content,
                    segment_type="text",
                    start_pos=buf.start_pos,
                    end_pos=nxt.end_pos,
                    metadata={**buf.metadata, **nxt.metadata}
                )
            else:
                merged.append(buf)
                buf = nxt
        merged.append(buf)

        return merged



class DocumentSegmenter:
    """Main document segmenter class"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.segmenter = self._initialize_segmenter()
        
    def _initialize_segmenter(self) -> BaseSegmenter:
        """Initialize based on configuration分段器"""
        method = self.config.get("split_method", "fixed").lower()
        max_length = self.config.get("max_segment_length", 1000)
        overlap_ratio = self.config.get("overlap_ratio", 0.1)
        
        if method == "fixed":
            return FixedLengthSegmenter(max_length, overlap_ratio)
        elif method == "semantic":
            similarity_threshold = self.config.get("similarity_threshold", 0.7)
            return SemanticSegmenter(
                max_length=max_length,
                similarity_threshold=similarity_threshold
            )
        elif method == "adaptive" or method == "hybrid":
            return HybridSegmenter(max_length, overlap_ratio)
        else:
            raise ValueError(f"不支持的Segmentation method: {method}")
    
    def segment_document(self, document: str, metadata: Optional[Dict] = None) -> List[DocumentSegment]:
        """分段文档"""
        logger.info(f"Starting document segmentation，length: {len(document)} characters")
        segments = self.segmenter.segment(document, metadata)
        logger.info(f"Segmentation completed，Generate {len(segments)} segments")
        return segments
    
    def get_segment_statistics(self, segments: List[DocumentSegment]) -> Dict[str, Any]:
        """获取分段统计信息"""
        if not segments:
            return {}
        
        lengths = [len(seg.content) for seg in segments]
        types = [seg.segment_type for seg in segments]
        
        stats = {
            "total_segments": len(segments),
            "avg_length": np.mean(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "std_length": np.std(lengths),
            "type_distribution": {t: types.count(t) for t in set(types)}
        }
        
        return stats


# 在 segmentation.py 批处理 CLI 同一文件中，替换/扩展 _pick_text，并新增 markdown 构造函数
def _get_ci(d: dict, key: str):
    """case-insensitive get: 在字典 d 中按不区分大小写查找 key，返回 (value, actual_key) 或 (None, None)"""
    key_l = key.lower()
    for k, v in d.items():
        if isinstance(k, str) and k.lower() == key_l:
            return v, k
    return None, None

def _rows_to_markdown_table(rows):
    if not isinstance(rows, list) or not rows:
        return None
    width = max(len(r) for r in rows if isinstance(r, list))
    norm = [(list(r) + [""] * (width - len(r))) if isinstance(r, list) else [""] * width for r in rows]
    header = norm[0] if norm else []
    if not any(header):
        header = [f"col_{i}" for i in range(width)]
        norm = [header] + norm
    sep = ["---"] * width
    lines = [
        "| " + " | ".join(str(x) for x in header) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for r in norm[1:]:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(lines)

def _pick_text(record: dict, candidate_keys: Tuple[str, ...]) -> Tuple[Optional[str], Optional[str]]:
    # 1) 先按候选键不区分大小写匹配（覆盖 document / Document / documentText / DOCUMENT 等）
    # 但跳过空字符串，因为有些数据集（如TatQA）的document_text为空但有其他内容源
    found_empty_document_text = False
    for k in candidate_keys:
        v, real_k = _get_ci(record, k)
        if isinstance(v, str):
            if v.strip():
                return v, real_k
            elif k.lower() in ("document_text", "document"):
                found_empty_document_text = True

    # 2) 常见变体（驼峰/下划线）- 同样跳过空字符串
    for k in ["document_text", "documentText", "doc_text", "docText", "document"]:
        v, real_k = _get_ci(record, k)
        if isinstance(v, str):
            if v.strip():
                return v, real_k
            else:
                found_empty_document_text = True

    # 3) 提取 paragraphs：既支持 list[str] 也支持 list[{"text": ...}] 或 {"Text": ...}
    # 支持直接在record中或在metadata中
    paras = record.get("paragraphs")
    if not paras and "metadata" in record:
        paras = record["metadata"].get("paragraphs")
    
    para_text = None
    if isinstance(paras, list):
        buf = []
        for it in paras:
            if isinstance(it, str) and it.strip():
                buf.append(it.strip())
            elif isinstance(it, dict):
                # 支持多种文本字段名
                for tk in ("text", "Text", "content", "Content"):
                    tv = it.get(tk)
                    if isinstance(tv, str) and tv.strip():
                        buf.append(tv.strip()); break
        if buf:
            para_text = "\n\n".join(buf)

    # 4) TAT-QA/DocFinQA 表格：{"table": {"table": [[...]]}} 或 {"table": [[...]]}
    # 支持直接在record中或在metadata中
    tbl = record.get("table")
    if not tbl and "metadata" in record:
        tbl = record["metadata"].get("table")
    
    rows = None
    if isinstance(tbl, dict) and isinstance(tbl.get("table"), list):
        rows = tbl["table"]
    elif isinstance(tbl, list):
        rows = tbl
    table_md = _rows_to_markdown_table(rows) if rows else None

    # 5) context 兜底（大小写兼容）
    # 支持直接在record中或在metadata中
    ctx, real_ctx = _get_ci(record, "context")
    if not (isinstance(ctx, str) and ctx.strip()):
        if "metadata" in record:
            ctx, real_ctx = _get_ci(record["metadata"], "context")

    # 6) question 兜底
    # 支持直接在record中或在metadata中
    q, real_q = _get_ci(record, "question")
    if not (isinstance(q, str) and q.strip()):
        if "metadata" in record:
            q, real_q = _get_ci(record["metadata"], "question")

    # 7) 组装优先级：paragraphs + table + document/document_text + context + question
    # 对于 TatQA 等数据集，paragraphs 和 table 是主要内容源
    parts = []
    used = []
    
    # 优先使用 paragraphs（如果存在且非空）
    if para_text:
        parts.append(para_text)
        used.append("paragraphs")
    
    # 添加表格（如果存在）
    if table_md:
        parts.append(table_md)
        used.append("table(markdown)")
    
    # 如果上面没匹配到 document/document_text，尝试收集
    if not parts or not any("document" in u for u in used):
        for k in ["document", "document_text", "documentText"]:
            v, real_k = _get_ci(record, k)
            if isinstance(v, str) and v.strip():
                parts.append(v)
                used.append(real_k)
                break
    
    # 添加 context（如果存在且前面没有足够内容）
    if ctx and len(parts) < 2:
        parts.append(ctx)
        used.append(real_ctx)
    
    # 最后兜底：question（只在没有其他内容时使用）
    if q and not parts:
        parts.append(q)
        used.append(real_q)

    # 如果只有 paragraphs 但内容为空，直接返回 paragraphs
    if para_text and not table_md:
        return para_text, "paragraphs"
    
    # 如果只有 table 但没有其他内容
    if table_md and not para_text:
        return table_md, "table(markdown)"
    
    # 组合多个部分
    if parts:
        return "\n\n".join(parts), "+".join(used) if used else "composed"

    return None, None

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def run_batch():
    parser = argparse.ArgumentParser("Batch segmenter for data/*/**/train.jsonl → data/segmented_data mirror")
    parser.add_argument("--data-root", default="data", help="根目录，递归寻找 train.jsonl")
    parser.add_argument("--out-root", default="data/segmented_data", help="输出镜像根目录")
    parser.add_argument("--glob-name", default="train.jsonl", help="目标文件名（默认只处理 train.jsonl）")
    # 分段配置（复用 DocumentSegmenter 的配置键）
    parser.add_argument("--split-method", default="hybrid", choices=["fixed", "semantic", "hybrid", "adaptive"])
    parser.add_argument("--max-segment-length", type=int, default=1000)
    parser.add_argument("--overlap-ratio", type=float, default=0.1)
    parser.add_argument("--similarity-threshold", type=float, default=0.7, help="用于 semantic")
    parser.add_argument("--semantic-breaks", action="store_true", help="hybrid 中文本切片时启用语义断点")
    parser.add_argument("--semantic-device", default="cuda", help="sentence-transformers 设备")
    # 输入文本字段候选
    parser.add_argument(
        "--text-keys",
        default="document,text,content,context,passage,article,body,report,raw_text",
        help="逗号分隔的候选字段名，按顺序匹配"
    )
    parser.add_argument("--keep-input", action="store_true", help="在输出中保留原始输入 record（可能很大）")
    parser.add_argument("--fail-fast", action="store_true", help="遇到解析/分段错误立即退出")
    args = parser.parse_args()

    data_root = Path(args.data_root).resolve()
    out_root = Path(args.out_root).resolve()
    candidate_keys = tuple([k.strip() for k in args.text_keys.split(",") if k.strip()])

    # 构造分段器
    seg_config = {
        "split_method": args.split_method,
        "max_segment_length": args.max_segment_length,
        "overlap_ratio": args.overlap_ratio,
        "similarity_threshold": args.similarity_threshold,
        "semantic_breaks": args.semantic_breaks,
        "semantic_device": args.semantic_device,
    }
    segmenter = DocumentSegmenter(seg_config)

    # 遍历所有 train.jsonl
    total_files = 0
    for root, _, files in os.walk(data_root):
        if args.glob_name not in files:
            continue

        in_path = Path(root) / args.glob_name
        rel_dir = Path(os.path.relpath(root, data_root))  # 相对 data_root 的路径
        mirror_dir = out_root / rel_dir
        _ensure_dir(mirror_dir)

        out_path = mirror_dir / args.glob_name.replace(".jsonl", ".segmented.jsonl")
        stats_path = mirror_dir / args.glob_name.replace(".jsonl", ".segmented.stats.json")

        print(f"[SEG] {in_path} → {out_path}")

        n_lines = 0
        n_ok = 0
        n_skip = 0
        n_err = 0
        total_segments = 0
        example_keys = set()

        with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
            for lineno, line in enumerate(fin, start=1):
                line = line.strip()
                if not line:
                    n_skip += 1
                    continue

                n_lines += 1
                try:
                    record = json.loads(line)
                except Exception as e:
                    n_err += 1
                    if args.fail_fast:
                        raise
                    # 写入错误占位（可选）
                    err_obj = {
                        "source": {
                            "path": str(in_path.relative_to(data_root)),
                            "line_no": lineno,
                        },
                        "error": f"JSONDecodeError: {str(e)}"
                    }
                    fout.write(json.dumps(err_obj, ensure_ascii=False) + "\n")
                    continue

                text, used_key = _pick_text(record, candidate_keys)
                if used_key:
                    example_keys.add(used_key)

                if not text:
                    n_skip += 1
                    warn_obj = {
                        "source": {
                            "path": str(in_path.relative_to(data_root)),
                            "line_no": lineno,
                            "id": record.get("id")
                        },
                        "warn": "no_text_field_found",
                        "tried_keys": candidate_keys
                    }
                    fout.write(json.dumps(warn_obj, ensure_ascii=False) + "\n")
                    continue

                try:
                    meta = {
                        "source_relpath": str(in_path.relative_to(data_root)),
                        "line_no": lineno,
                        "record_id": record.get("id"),
                        "used_key": used_key
                    }
                    segs = segmenter.segment_document(text, metadata=meta)
                    total_segments += len(segs)

                    out_obj = {
                        "source": meta,
                        "segments": [asdict(s) for s in segs],
                        "segmentation_config": {
                            "method": args.split_method,
                            "max_segment_length": args.max_segment_length,
                            "overlap_ratio": args.overlap_ratio,
                            "semantic_breaks": args.semantic_breaks
                        }
                    }
                    if args.keep_input:
                        out_obj["input"] = record

                    fout.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                    n_ok += 1

                except Exception as e:
                    n_err += 1
                    if args.fail_fast:
                        raise
                    err_obj = {
                        "source": meta,
                        "error": f"SegmentationError: {str(e)}"
                    }
                    fout.write(json.dumps(err_obj, ensure_ascii=False) + "\n")

        # 写文件级别统计
        stats = {
            "input_file": str(in_path),
            "output_file": str(out_path),
            "lines_total": n_lines,
            "lines_segmented": n_ok,
            "lines_skipped_no_text": n_skip,
            "lines_error": n_err,
            "avg_segments_per_line": (total_segments / n_ok) if n_ok else 0.0,
            "used_text_keys": sorted(list(example_keys)),
            "segmenter_config": seg_config
        }
        with stats_path.open("w", encoding="utf-8") as fstats:
            json.dump(stats, fstats, ensure_ascii=False, indent=2)

        total_files += 1

    print(f"[DONE] processed files: {total_files}  (root={data_root})")


if __name__ == "__main__":
    run_batch()