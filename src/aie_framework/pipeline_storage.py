"""
AIE Pipeline Storage Manager - 中间数据存档管理

提供AIE pipeline各阶段的数据存档和缓存功能，优化性能和可重现性。
"""

import json
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import time

from .segmentation import DocumentSegment
from .retrieval import RetrievalResult  
from .summarization import SummaryResult
from .extraction import ExtractionResult

logger = logging.getLogger(__name__)


@dataclass
class PipelineCache:
    """Pipeline缓存配置"""
    enabled: bool = True
    base_dir: str = "data/pipeline_cache"
    segments_dir: str = "segments"
    retrieval_dir: str = "retrieval" 
    summary_dir: str = "summaries"
    extraction_dir: str = "extractions"
    ttl_hours: int = 24 * 7  # 缓存保留7天


class PipelineStorageManager:
    """AIE Pipeline存档管理器"""
    
    def __init__(self, cache_config: PipelineCache):
        self.config = cache_config
        self.base_path = Path(cache_config.base_dir)
        
        if cache_config.enabled:
            # 创建缓存目录结构
            self.segments_path = self.base_path / cache_config.segments_dir
            self.retrieval_path = self.base_path / cache_config.retrieval_dir
            self.summary_path = self.base_path / cache_config.summary_dir
            self.extraction_path = self.base_path / cache_config.extraction_dir
            
            for path in [self.segments_path, self.retrieval_path, 
                        self.summary_path, self.extraction_path]:
                path.mkdir(parents=True, exist_ok=True)
                
        logger.info(f"PipelineStorageManager initialized: enabled={cache_config.enabled}")
    
    def _generate_cache_key(self, content: str, config: Dict[str, Any] = None) -> str:
        """生成缓存key"""
        payload = {
            "content_hash": hashlib.sha256(content.encode('utf-8')).hexdigest()[:16],
            "config": config or {}
        }
        key_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:16]
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """检查缓存是否仍然有效"""
        if not cache_file.exists():
            return False
            
        # 检查TTL
        file_age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        return file_age_hours < self.config.ttl_hours
    
    # =============== Segmentation Cache ===============
    
    def save_segments(self, document_text: str, segments: List[DocumentSegment], 
                     config: Dict[str, Any]) -> Optional[str]:
        """保存分段结果"""
        if not self.config.enabled:
            return None
            
        try:
            cache_key = self._generate_cache_key(document_text, config)
            cache_file = self.segments_path / f"{cache_key}.jsonl"
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                # 保存元数据
                metadata = {
                    "timestamp": time.time(),
                    "config": config,
                    "document_length": len(document_text),
                    "segments_count": len(segments)
                }
                f.write(json.dumps({"_metadata": metadata}) + '\n')
                
                # 保存分段数据
                for seg in segments:
                    seg_dict = {
                        "id": seg.id,
                        "content": seg.content,
                        "segment_type": seg.segment_type,
                        "start_pos": seg.start_pos,
                        "end_pos": seg.end_pos,
                        "metadata": seg.metadata
                    }
                    f.write(json.dumps(seg_dict, ensure_ascii=False) + '\n')
            
            logger.debug(f"Saved {len(segments)} segments to {cache_file}")
            return cache_key
            
        except Exception as e:
            logger.warning(f"Failed to save segments: {e}")
            return None
    
    def load_segments(self, document_text: str, config: Dict[str, Any]) -> Optional[List[DocumentSegment]]:
        """加载分段结果"""
        if not self.config.enabled:
            return None
            
        try:
            cache_key = self._generate_cache_key(document_text, config)
            cache_file = self.segments_path / f"{cache_key}.jsonl"
            
            if not self._is_cache_valid(cache_file):
                return None
            
            segments = []
            with open(cache_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if not line.strip():
                        continue
                    
                    data = json.loads(line)
                    
                    # 跳过元数据行
                    if i == 0 and "_metadata" in data:
                        continue
                    
                    # 重建DocumentSegment对象
                    segment = DocumentSegment(
                        id=data["id"],
                        content=data["content"],
                        segment_type=data["segment_type"],
                        start_pos=data["start_pos"],
                        end_pos=data["end_pos"],
                        metadata=data["metadata"]
                    )
                    segments.append(segment)
            
            logger.debug(f"Loaded {len(segments)} segments from cache")
            return segments
            
        except Exception as e:
            logger.warning(f"Failed to load segments: {e}")
            return None
    
    # =============== Retrieval Cache ===============
    
    def save_retrieval_results(self, query: str, segments: List[DocumentSegment], 
                              results: List[RetrievalResult], config: Dict[str, Any]) -> Optional[str]:
        """保存检索结果"""
        if not self.config.enabled:
            return None
            
        try:
            # 构建缓存key（包含查询和分段内容）
            content = query + "|" + "|".join([seg.content[:100] for seg in segments])
            cache_key = self._generate_cache_key(content, config)
            cache_file = self.retrieval_path / f"{cache_key}.json"
            
            cache_data = {
                "_metadata": {
                    "timestamp": time.time(),
                    "query": query,
                    "config": config,
                    "segments_count": len(segments),
                    "results_count": len(results)
                },
                "results": [
                    {
                        "rank": r.rank,
                        "score": r.score,
                        "segment_id": r.segment.id,
                        "segment_content": r.segment.content,
                        "segment_type": r.segment.segment_type,
                        "metadata": r.metadata
                    }
                    for r in results
                ]
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"Saved {len(results)} retrieval results to {cache_file}")
            return cache_key
            
        except Exception as e:
            logger.warning(f"Failed to save retrieval results: {e}")
            return None
    
    def load_retrieval_results(self, query: str, segments: List[DocumentSegment], 
                              config: Dict[str, Any]) -> Optional[List[RetrievalResult]]:
        """加载检索结果"""
        if not self.config.enabled:
            return None
            
        try:
            content = query + "|" + "|".join([seg.content[:100] for seg in segments])
            cache_key = self._generate_cache_key(content, config)
            cache_file = self.retrieval_path / f"{cache_key}.json"
            
            if not self._is_cache_valid(cache_file):
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # 重建RetrievalResult对象
            results = []
            for r_data in cache_data["results"]:
                # 重建segment对象
                segment = DocumentSegment(
                    id=r_data["segment_id"],
                    content=r_data["segment_content"],
                    segment_type=r_data["segment_type"],
                    start_pos=0,  # 简化，实际使用中可以从segments中查找
                    end_pos=len(r_data["segment_content"]),
                    metadata={}
                )
                
                result = RetrievalResult(
                    rank=r_data["rank"],
                    score=r_data["score"],
                    segment=segment,
                    metadata=r_data["metadata"]
                )
                results.append(result)
            
            logger.debug(f"Loaded {len(results)} retrieval results from cache")
            return results
            
        except Exception as e:
            logger.warning(f"Failed to load retrieval results: {e}")
            return None
    
    # =============== Cache Management ===============
    
    def cleanup_expired_cache(self):
        """清理过期缓存"""
        if not self.config.enabled:
            return
            
        cleaned_count = 0
        for cache_dir in [self.segments_path, self.retrieval_path, 
                         self.summary_path, self.extraction_path]:
            for cache_file in cache_dir.glob("*"):
                if not self._is_cache_valid(cache_file):
                    try:
                        cache_file.unlink()
                        cleaned_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to delete {cache_file}: {e}")
        
        logger.info(f"Cleaned up {cleaned_count} expired cache files")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if not self.config.enabled:
            return {"enabled": False}
            
        stats = {"enabled": True}
        
        for name, path in [
            ("segments", self.segments_path),
            ("retrieval", self.retrieval_path), 
            ("summaries", self.summary_path),
            ("extractions", self.extraction_path)
        ]:
            files = list(path.glob("*"))
            valid_files = [f for f in files if self._is_cache_valid(f)]
            
            total_size = sum(f.stat().st_size for f in files) / (1024**2)  # MB
            
            stats[name] = {
                "total_files": len(files),
                "valid_files": len(valid_files),
                "total_size_mb": round(total_size, 2)
            }
        
        return stats


# =============== Integration with AIEPipeline ===============

def create_enhanced_pipeline_with_storage(config: Dict[str, Any], llm_interface, 
                                         cache_config: Optional[PipelineCache] = None):
    """创建带存档功能的增强Pipeline"""
    from .pipeline import AIEPipeline
    
    # 添加存档管理器
    cache_config = cache_config or PipelineCache()
    storage_manager = PipelineStorageManager(cache_config)
    
    # 扩展原有pipeline配置
    enhanced_config = {
        **config,
        "storage": {
            "manager": storage_manager,
            "cache_segments": True,
            "cache_retrieval": True,
            "cache_summaries": False,  # 摘要通常动态性较强
            "cache_extractions": False  # 提取结果通常需要实时
        }
    }
    
    return AIEPipeline(enhanced_config, llm_interface)


# =============== Usage Example ===============

def example_usage():
    """使用示例"""
    
    # 1. 创建缓存配置
    cache_config = PipelineCache(
        enabled=True,
        base_dir="data/pipeline_cache",
        ttl_hours=24 * 7  # 缓存7天
    )
    
    # 2. 创建存档管理器
    storage = PipelineStorageManager(cache_config)
    
    # 3. 在pipeline中使用
    # 在segmentation阶段之后：
    # cache_key = storage.save_segments(document_text, segments, seg_config)
    
    # 在pipeline开始时尝试加载缓存：
    # cached_segments = storage.load_segments(document_text, seg_config)
    # if cached_segments:
    #     logger.info("Using cached segments")
    #     segments = cached_segments
    # else:
    #     segments = segmenter.segment_document(document_text)
    #     storage.save_segments(document_text, segments, seg_config)
    
    # 4. 定期清理缓存
    # storage.cleanup_expired_cache()
    
    # 5. 监控缓存状态
    # stats = storage.get_cache_stats()
    # logger.info(f"Cache stats: {stats}")


if __name__ == "__main__":
    example_usage()
