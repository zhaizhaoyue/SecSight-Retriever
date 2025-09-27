"""
FINE Dataset Processing Module (Enhanced)

Process Financial Reports Numerical Extraction (FINE) dataset

Enhancements:
- Deterministic, configurable random seed and shuffling
- Optional stratified and/or group-aware splitting to reduce leakage
- Lightweight text/query normalization and flexible filtering hooks
- De-duplication controls (by document_id and/or normalized query)
- Optional disk caching keyed by data files + processing config
- Utilities: iterators, export splits, extended statistics
"""

import os
import json
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Iterable, Set
from dataclasses import dataclass
from pathlib import Path
import random
import statistics

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class FINEDataPoint:
    """Single data point in FINE dataset"""
    document_id: str
    document_text: str
    query: str
    target_value: Any
    target_type: str
    metadata: Dict[str, Any]
    
    def __str__(self) -> str:
        return f"FINEDataPoint(id={self.document_id}, query='{self.query[:50]}...', target={self.target_value})"


@dataclass
class FINEBatch:
    """FINE dataset batch"""
    data_points: List[FINEDataPoint]
    split: str  # train, val, test
    
    def __len__(self) -> int:
        return len(self.data_points)
    
    def __getitem__(self, index: int) -> FINEDataPoint:
        return self.data_points[index]


class FINEDataset:
    """FINE dataset processing class"""
    
    def __init__(self, data_path: str, config: Dict[str, Any]):
        self.data_path = Path(data_path)
        self.config = config or {}
        
        # data split比例
        self.train_split = float(self.config.get("train_split", 0.8))
        self.val_split = float(self.config.get("val_split", 0.1))
        self.test_split = float(self.config.get("test_split", 0.1))
        # ensure ratios are coherent
        s = self.train_split + self.val_split + self.test_split
        if s <= 0:
            self.train_split, self.val_split, self.test_split = 0.8, 0.1, 0.1
        else:
            self.train_split /= s
            self.val_split /= s
            self.test_split /= s

        # randomness / determinism
        self.seed = int(self.config.get("seed", 42))
        self.shuffle = bool(self.config.get("shuffle", True))
        self.random_state = random.Random(self.seed)

        # splitting strategy
        # stratify by a field among: "target_type", "company", "year"; or None
        self.stratify_field = self.config.get("stratify_field")
        # group-aware split to reduce leakage: e.g., group by "company" or "document_id"
        self.group_field = self.config.get("group_field")

        # processing toggles
        self.enable_normalize_text = bool(self.config.get("normalize_text", True))
        self.enable_normalize_query = bool(self.config.get("normalize_query", True))
        self.enable_deduplicate = bool(self.config.get("deduplicate", True))

        # filtering knobs
        self.min_doc_len = self.config.get("min_document_length")
        self.max_doc_len = self.config.get("max_document_length")
        self.allowed_target_types = set(self.config.get("allowed_target_types", []) or [])
        self.allowed_years = set(self.config.get("allowed_years", []) or [])

        # on-disk cache
        self.disk_cache_enabled = bool(self.config.get("disk_cache", False))
        self.cache_dir = Path(self.config.get("cache_dir", self.data_path.parent / ".cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据缓存
        self._train_data = None
        self._val_data = None
        self._test_data = None
        self._full_data = None
        
        logger.info(f"初始化FINE数据集，路径: {self.data_path}")

    # --------------- normalization & filtering ---------------
    @staticmethod
    def _basic_whitespace_norm(text: str) -> str:
        if not isinstance(text, str):
            return ""
        # collapse whitespace and normalize quotes
        out = " ".join(text.split())
        out = out.replace("\u00a0", " ").replace("\u2019", "'")
        return out

    def _normalize_text(self, text: str) -> str:
        if not self.enable_normalize_text:
            return text or ""
        return self._basic_whitespace_norm(text or "")

    def _normalize_query(self, query: str) -> str:
        if not self.enable_normalize_query:
            return query or ""
        q = self._basic_whitespace_norm(query or "")
        # simple lower for comparability (do not change semantics)
        return q.lower()

    def _filter_data_point(self, dp: "FINEDataPoint") -> bool:
        """Return True to keep the datapoint."""
        if self.min_doc_len is not None and len(dp.document_text or "") < int(self.min_doc_len):
            return False
        if self.max_doc_len is not None and len(dp.document_text or "") > int(self.max_doc_len):
            return False
        if self.allowed_target_types and dp.target_type not in self.allowed_target_types:
            return False
        if self.allowed_years:
            y = str(dp.metadata.get("year", ""))
            if y and y not in self.allowed_years:
                return False
        return True

    # --------------- caching ---------------
    def _cache_key(self, files: List[Path]) -> str:
        payload = {
            "files": [f.resolve().as_posix() for f in sorted(files)],
            "train": self.train_split,
            "val": self.val_split,
            "test": self.test_split,
            "seed": self.seed,
            "normalize_text": self.enable_normalize_text,
            "normalize_query": self.enable_normalize_query,
            "deduplicate": self.enable_deduplicate,
            "min_len": self.min_doc_len,
            "max_len": self.max_doc_len,
            "allowed_types": sorted(self.allowed_target_types),
            "allowed_years": sorted(self.allowed_years),
        }
        s = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"fine_dataset_cache_{key}.jsonl"
    
    def _load_json_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """加载JSON文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"successful加载 {file_path}，包含 {len(data)} 条记录")
            return data
        except Exception as e:
            logger.error(f"加载JSON文件failed {file_path}: {e}")
            return []
    
    def _load_csv_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """加载CSV文件"""
        try:
            df = pd.read_csv(file_path)
            data = df.to_dict('records')
            logger.info(f"successful加载 {file_path}，包含 {len(data)} 条记录")
            return data
        except Exception as e:
            logger.error(f"加载CSV文件failed {file_path}: {e}")
            return []
    
    def _create_data_point(self, raw_data: Dict[str, Any], index: int) -> FINEDataPoint:
        """从原始数据创建数据点"""
        dp = FINEDataPoint(
            document_id=raw_data.get("document_id", f"doc_{index:06d}"),
            document_text=raw_data.get("document_text", raw_data.get("text", "")),
            query=raw_data.get("query", raw_data.get("question", "")),
            target_value=raw_data.get("target_value", raw_data.get("answer", raw_data.get("value"))),
            target_type=raw_data.get("target_type", raw_data.get("type", "number")),
            metadata={
                "source_file": raw_data.get("source_file", ""),
                "company": raw_data.get("company", ""),
                "year": raw_data.get("year", ""),
                "report_type": raw_data.get("report_type", ""),
                **{k: v for k, v in raw_data.items() if k not in 
                   ["document_id", "document_text", "query", "target_value", "target_type"]}
            }
        )
        # lightweight normalization
        dp = FINEDataPoint(
            document_id=str(dp.document_id),
            document_text=self._normalize_text(dp.document_text),
            query=self._normalize_query(dp.query),
            target_value=dp.target_value,
            target_type=str(dp.target_type or "number"),
            metadata=dp.metadata or {}
        )
        return dp
    
    def load_data(self) -> List[FINEDataPoint]:
        """加载完整数据集"""
        if self._full_data is not None:
            return self._full_data
        
        logger.info(f"开始加载FINE数据集...")
        
        all_data = []
        
        # 查找数据文件
        data_files = []
        if self.data_path.is_file():
            data_files = [self.data_path]
        elif self.data_path.is_dir():
            # 查找目录中的数据文件
            for pattern in ["*.json", "*.jsonl", "*.csv"]:
                data_files.extend(self.data_path.glob(pattern))
        
        if not data_files:
            logger.warning(f"在 {self.data_path} 中未找到数据文件")
            return []

        # try disk cache
        cache_key = self._cache_key(data_files)
        cache_fp = self._cache_path(cache_key)
        if self.disk_cache_enabled and cache_fp.exists():
            try:
                logger.info("使用磁盘缓存: %s", cache_fp.as_posix())
                cached: List[FINEDataPoint] = []
                with open(cache_fp, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        obj = json.loads(line)
                        cached.append(FINEDataPoint(**obj))
                self._full_data = cached
                return cached
            except Exception as e:
                logger.warning("读取缓存失败，将重新构建: %s", e)
        
        # 加载所有数据文件
        for file_path in data_files:
            if file_path.suffix.lower() == '.json':
                raw_data_list = self._load_json_file(file_path)
            elif file_path.suffix.lower() == '.jsonl':
                # processingJSONL文件
                raw_data_list = []
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                raw_data_list.append(json.loads(line))
                except Exception as e:
                    logger.error(f"加载JSONL文件failed {file_path}: {e}")
                    continue
            elif file_path.suffix.lower() == '.csv':
                raw_data_list = self._load_csv_file(file_path)
            else:
                logger.warning(f"不支持的文件格式: {file_path}")
                continue
            
            # 转换为数据点
            for i, raw_data in enumerate(raw_data_list):
                try:
                    data_point = self._create_data_point(raw_data, len(all_data))
                    if not self._filter_data_point(data_point):
                        continue
                    all_data.append(data_point)
                except Exception as e:
                    logger.warning(f"创建数据点failed (file: {file_path}, index: {i}): {e}")
        
        # de-duplicate
        if self.enable_deduplicate and all_data:
            seen: Set[Tuple[str, str]] = set()
            deduped: List[FINEDataPoint] = []
            for dp in all_data:
                key = (dp.document_id, dp.query)
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(dp)
            if len(deduped) != len(all_data):
                logger.info("去重: %d -> %d", len(all_data), len(deduped))
            all_data = deduped

        # optional shuffle for training robustness (deterministic)
        if self.shuffle:
            self.random_state.shuffle(all_data)

        # write cache
        if self.disk_cache_enabled:
            try:
                with open(cache_fp, "w", encoding="utf-8") as f:
                    for dp in all_data:
                        f.write(json.dumps(dp.__dict__, ensure_ascii=False) + "\n")
                logger.info("已写入磁盘缓存: %s", cache_fp.as_posix())
            except Exception as e:
                logger.warning("写入缓存失败: %s", e)

        self._full_data = all_data
        logger.info(f"successful加载 {len(all_data)} 个数据点")
        
        return all_data
    
    def _split_data(self, data: List[FINEDataPoint]) -> Tuple[List[FINEDataPoint], List[FINEDataPoint], List[FINEDataPoint]]:
        """Split dataset with optional stratification and grouping (deterministic)."""
        if not data:
            return [], [], []

        indices = list(range(len(data)))

        # Helper: group mapping
        def group_id(i: int) -> str:
            if not self.group_field:
                return str(i)
            if self.group_field == "document_id":
                return str(data[i].document_id)
            if self.group_field == "company":
                return str(data[i].metadata.get("company", ""))
            if self.group_field == "year":
                return str(data[i].metadata.get("year", ""))
            return str(getattr(data[i], self.group_field, ""))

        # Helper: stratify label
        def strata_label(i: int) -> str:
            fld = self.stratify_field
            if not fld:
                return "__ALL__"
            if fld == "target_type":
                return str(data[i].target_type)
            if fld == "company":
                return str(data[i].metadata.get("company", ""))
            if fld == "year":
                return str(data[i].metadata.get("year", ""))
            return str(getattr(data[i], fld, ""))

        # Build buckets either by strata or all
        buckets: Dict[str, List[int]] = {}
        for idx in indices:
            label = strata_label(idx)
            buckets.setdefault(label, []).append(idx)

        # Within each bucket, optionally group by group_id so samples from same group go to same split
        train_idx: List[int] = []
        val_idx: List[int] = []
        test_idx: List[int] = []

        for label, idxs in buckets.items():
            # group mapping
            if self.group_field:
                gid2idxs: Dict[str, List[int]] = {}
                for i in idxs:
                    gid2idxs.setdefault(group_id(i), []).append(i)
                groups = list(gid2idxs.keys())
                self.random_state.shuffle(groups)
                n = len(groups)
                n_train = int(round(n * self.train_split))
                n_val = int(round(n * self.val_split))
                n_train = max(0, min(n, n_train))
                n_val = max(0, min(n - n_train, n_val))
                n_test = n - n_train - n_val
                gs_train = set(groups[:n_train])
                gs_val = set(groups[n_train:n_train + n_val])
                for g, gidxs in gid2idxs.items():
                    if g in gs_train:
                        train_idx.extend(gidxs)
                    elif g in gs_val:
                        val_idx.extend(gidxs)
                    else:
                        test_idx.extend(gidxs)
            else:
                # simple stratified split within bucket
                arr = list(idxs)
                self.random_state.shuffle(arr)
                n = len(arr)
                n_train = int(round(n * self.train_split))
                n_val = int(round(n * self.val_split))
                n_train = max(0, min(n, n_train))
                n_val = max(0, min(n - n_train, n_val))
                train_idx.extend(arr[:n_train])
                val_idx.extend(arr[n_train:n_train + n_val])
                test_idx.extend(arr[n_train + n_val:])

        # map back
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        test_data = [data[i] for i in test_idx]

        logger.info(
            "data split: 训练集 %d, 验证集 %d, 测试集 %d (stratify=%s, group=%s)",
            len(train_data), len(val_data), len(test_data),
            str(self.stratify_field), str(self.group_field)
        )
        return train_data, val_data, test_data
    
    def get_train_data(self) -> FINEBatch:
        """Get training set"""
        if self._train_data is None:
            full_data = self.load_data()
            train_data, val_data, test_data = self._split_data(full_data)
            self._train_data = train_data
            self._val_data = val_data
            self._test_data = test_data
        
        return FINEBatch(self._train_data, "train")
    
    def get_val_data(self) -> FINEBatch:
        """Get validation set"""
        if self._val_data is None:
            full_data = self.load_data()
            train_data, val_data, test_data = self._split_data(full_data)
            self._train_data = train_data
            self._val_data = val_data
            self._test_data = test_data
        
        return FINEBatch(self._val_data, "val")
    
    def get_test_data(self) -> FINEBatch:
        """Get test set"""
        if self._test_data is None:
            full_data = self.load_data()
            train_data, val_data, test_data = self._split_data(full_data)
            self._train_data = train_data
            self._val_data = val_data
            self._test_data = test_data
        
        return FINEBatch(self._test_data, "test")
    
    def get_sample_data(self, n: int = 10, split: str = "train") -> FINEBatch:
        """Get sample data"""
        if split == "train":
            batch = self.get_train_data()
        elif split == "val":
            batch = self.get_val_data()
        elif split == "test":
            batch = self.get_test_data()
        else:
            raise ValueError(f"不支持的data split: {split}")
        
        sample_data = batch.data_points[:min(n, len(batch.data_points))]
        return FINEBatch(sample_data, f"{split}_sample")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取数据集统计信息"""
        full_data = self.load_data()
        
        if not full_data:
            return {}
        
        # 基本统计
        total_count = len(full_data)
        
        # 文档length统计
        doc_lengths = [len(dp.document_text) for dp in full_data]
        
        # querylength统计
        query_lengths = [len(dp.query) for dp in full_data]
        
        # 目标类型分布
        target_types = [dp.target_type for dp in full_data]
        type_distribution = {t: target_types.count(t) for t in set(target_types)}
        
        # 公司分布
        companies = [dp.metadata.get("company", "unknown") for dp in full_data]
        company_distribution = {c: companies.count(c) for c in set(companies)}
        
        # 年份分布
        years = [dp.metadata.get("year", "unknown") for dp in full_data]
        year_distribution = {y: years.count(y) for y in set(years)}
        
        # 更丰富的统计（稳健分位数）
        def _percentile(arr: List[int], p: float) -> int:
            if not arr:
                return 0
            k = max(0, min(len(arr) - 1, int(round((p / 100.0) * (len(arr) - 1)))))
            return sorted(arr)[k]
        
        stats = {
            "total_count": total_count,
            "document_length": {
                "min": min(doc_lengths),
                "max": max(doc_lengths),
                "avg": sum(doc_lengths) / len(doc_lengths),
                "median": sorted(doc_lengths)[len(doc_lengths) // 2],
                "p90": _percentile(doc_lengths, 90),
                "p95": _percentile(doc_lengths, 95)
            },
            "query_length": {
                "min": min(query_lengths),
                "max": max(query_lengths),
                "avg": sum(query_lengths) / len(query_lengths),
                "median": sorted(query_lengths)[len(query_lengths) // 2],
                "p90": _percentile(query_lengths, 90),
                "p95": _percentile(query_lengths, 95)
            },
            "target_type_distribution": type_distribution,
            "company_distribution": company_distribution,
            "year_distribution": year_distribution
        }
        
        return stats

    # --------------------- Utilities ---------------------
    def iter_data(self) -> Iterable[FINEDataPoint]:
        """Iterate over all datapoints without creating batches (after processing)."""
        for dp in self.load_data():
            yield dp

    def export_splits(self, out_dir: str) -> Dict[str, str]:
        """Export train/val/test to JSONL files. Returns file paths."""
        train = self.get_train_data().data_points
        val = self.get_val_data().data_points
        test = self.get_test_data().data_points

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        def _dump(name: str, dps: List[FINEDataPoint]) -> str:
            fp = out / f"{name}.jsonl"
            with open(fp, "w", encoding="utf-8") as f:
                for dp in dps:
                    f.write(json.dumps(dp.__dict__, ensure_ascii=False) + "\n")
            return fp.as_posix()

        return {
            "train": _dump("train", train),
            "val": _dump("val", val),
            "test": _dump("test", test)
        }
    
    def create_sample_dataset(self, output_path: str, n_samples: int = 100):
        """Create sample dataset用于测试"""
        logger.info(f"Create sample dataset，样本数: {n_samples}")
        
        # 创建模拟数据
        sample_data = []
        for i in range(n_samples):
            data_point = FINEDataPoint(
                document_id=f"sample_doc_{i:03d}",
                document_text=f"这是一份样本金融报告 {i}。公司在2023年实现营业收入{100 + i * 10}万元，净利润{20 + i * 2}万元。",
                query=f"2023年营业收入是多少？",
                target_value=100 + i * 10,
                target_type="number",
                metadata={
                    "company": f"Company_{i % 10}",
                    "year": "2023",
                    "report_type": "annual"
                }
            )
            sample_data.append(data_point)
        
        # 保存样本数据
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        sample_json = []
        for dp in sample_data:
            sample_json.append({
                "document_id": dp.document_id,
                "document_text": dp.document_text,
                "query": dp.query,
                "target_value": dp.target_value,
                "target_type": dp.target_type,
                **dp.metadata
            })
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sample_json, f, ensure_ascii=False, indent=2)
        
        logger.info(f"样本数据集已保存到: {output_path}")
        
        return sample_data
