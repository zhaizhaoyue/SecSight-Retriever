"""
Large Language Model Interface Module (Enhanced)

Support multiple LLM providers with advanced features:
- Intelligent retry with exponential backoff and circuit breaker
- Concurrent API calls with rate limiting and semaphore control
- Response caching and cost tracking
- Accurate token counting using tiktoken
- Performance monitoring and error analytics
- Streaming generation and batch processing optimization
"""

import os
import time
import asyncio
import hashlib
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import json
from functools import lru_cache
from pathlib import Path

import openai
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Optional dependencies for enhanced functionality
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    logging.warning("tiktoken not available, using approximate token counting")

logger = logging.getLogger(__name__)


# ============ Data Models ============

@dataclass
class APICallStats:
    """API调用统计"""
    total_calls: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    total_time: float = 0.0
    error_count: int = 0
    avg_response_time: float = 0.0
    last_call_time: Optional[float] = None
    
    def update(self, tokens: int, cost: float, elapsed_time: float, success: bool = True):
        """更新统计信息"""
        self.total_calls += 1
        self.total_tokens += tokens
        self.total_cost += cost
        self.total_time += elapsed_time
        if not success:
            self.error_count += 1
        self.avg_response_time = self.total_time / self.total_calls
        self.last_call_time = time.time()


@dataclass
class CircuitBreaker:
    """熔断器"""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    failure_count: int = 0
    last_failure_time: Optional[float] = None
    state: str = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    def call_failed(self):
        """记录调用失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def call_succeeded(self):
        """记录调用成功"""
        self.failure_count = 0
        self.state = "CLOSED"
    
    def can_call(self) -> bool:
        """是否可以调用"""
        if self.state == "CLOSED":
            return True
        elif self.state == "OPEN":
            if self.last_failure_time and time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return True
            return False
        else:  # HALF_OPEN
            return True


class ResponseCache:
    """响应缓存"""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._lock = threading.Lock()
    
    def _hash_key(self, prompt: str, **kwargs) -> str:
        """生成缓存key"""
        cache_data = {"prompt": prompt, **kwargs}
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def get(self, prompt: str, **kwargs) -> Optional[str]:
        """获取缓存"""
        key = self._hash_key(prompt, **kwargs)
        with self._lock:
            if key in self._cache:
                response, timestamp = self._cache[key]
                if time.time() - timestamp < self.ttl:
                    return response
                else:
                    del self._cache[key]
        return None
    
    def put(self, prompt: str, response: str, **kwargs):
        """设置缓存"""
        key = self._hash_key(prompt, **kwargs)
        with self._lock:
            if len(self._cache) >= self.max_size:
                # 删除最旧的条目
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
            self._cache[key] = (response, time.time())


# ============ Enhanced Base Class ============

class BaseLLM(ABC):
    """Enhanced base LLM class with monitoring and caching"""
    
    def __init__(self, **kwargs):
        self.stats = APICallStats()
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=kwargs.get("failure_threshold", 5),
            recovery_timeout=kwargs.get("recovery_timeout", 60.0)
        )
        self.cache = ResponseCache(
            max_size=kwargs.get("cache_size", 1000),
            ttl=kwargs.get("cache_ttl", 3600)
        ) if kwargs.get("enable_cache", True) else None
        
        # Rate limiting
        self.rate_limit = kwargs.get("rate_limit", 60)  # calls per minute
        self.call_times = []
        self._rate_lock = threading.Lock()
    
    def _check_rate_limit(self):
        """检查速率限制"""
        with self._rate_lock:
            now = time.time()
            # 移除1分钟前的调用记录
            self.call_times = [t for t in self.call_times if now - t < 60]
            
            if len(self.call_times) >= self.rate_limit:
                sleep_time = 60 - (now - self.call_times[0])
                if sleep_time > 0:
                    logger.info(f"Rate limit reached, sleeping for {sleep_time:.2f}s")
                    time.sleep(sleep_time)
            
            self.call_times.append(now)
    
    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """估算API调用成本（需要子类实现具体定价）"""
        return 0.0  # 基类返回0，子类重写
    
    @abstractmethod
    def _generate_impl(self, prompt: str, **kwargs) -> str:
        """实际的生成实现（由子类实现）"""
        pass
    
    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count tokens"""
        pass
    
    def generate(self, prompt: str, use_cache: bool = True, **kwargs) -> str:
        """Enhanced generation with caching, monitoring, and circuit breaker"""
        # 检查熔断器
        if not self.circuit_breaker.can_call():
            raise Exception("Circuit breaker is open")
        
        # 检查缓存
        if use_cache and self.cache:
            cached_response = self.cache.get(prompt, **kwargs)
            if cached_response:
                logger.debug("Cache hit for prompt")
                return cached_response
        
        # 检查速率限制
        self._check_rate_limit()
        
        start_time = time.time()
        success = False
        
        try:
            response = self._generate_impl(prompt, **kwargs)
            success = True
            
            # 更新统计
            input_tokens = self.count_tokens(prompt)
            output_tokens = self.count_tokens(response)
            cost = self._estimate_cost(input_tokens, output_tokens)
            elapsed_time = time.time() - start_time
            
            self.stats.update(input_tokens + output_tokens, cost, elapsed_time, success)
            self.circuit_breaker.call_succeeded()
            
            # 更新缓存
            if use_cache and self.cache:
                self.cache.put(prompt, response, **kwargs)
            
            return response
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            self.stats.update(0, 0, elapsed_time, success)
            self.circuit_breaker.call_failed()
            logger.error(f"Generation failed: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_calls": self.stats.total_calls,
            "total_tokens": self.stats.total_tokens,
            "total_cost": self.stats.total_cost,
            "avg_response_time": self.stats.avg_response_time,
            "error_rate": self.stats.error_count / max(1, self.stats.total_calls),
            "circuit_breaker_state": self.circuit_breaker.state
        }

# ============ Enhanced Provider Implementations ============

class DeepSeekLLM(BaseLLM):
    """Enhanced DeepSeek model interface with accurate token counting and cost tracking"""
    
    def __init__(self, model_name: str = "deepseek-chat", api_key: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        import openai
        self.model_name = model_name
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )
        self.default_params = {
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        
        # DeepSeek定价 (approximate, 需要根据实际API更新)
        self.pricing = {
            "input_cost_per_1k": 0.0014,  # $1.4/1M tokens
            "output_cost_per_1k": 0.0028  # $2.8/1M tokens
        }
        
        # 尝试使用tiktoken进行准确的token计数
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 encoding
            except Exception:
                logger.warning("Failed to load tiktoken encoder, using approximation")

    def _generate_impl(self, prompt: str, **kwargs) -> str:
        """实际的DeepSeek API调用"""
        params = {**self.default_params, **kwargs}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **params
        )
        return response.choices[0].message.content.strip()

    def count_tokens(self, text: str) -> int:
        """准确的token计数"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # 回退到近似计算
            return int(len(text.split()) * 1.3)
    
    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """估算API调用成本"""
        input_cost = (input_tokens / 1000) * self.pricing["input_cost_per_1k"]
        output_cost = (output_tokens / 1000) * self.pricing["output_cost_per_1k"]
        return input_cost + output_cost

class OpenAILLM(BaseLLM):
    """Enhanced OpenAI model interface with accurate token counting and cost tracking"""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo", api_key: Optional[str] = None, base_url: Optional[str] = None, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.client = openai.OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY"),
            base_url=base_url
        )
        self.default_params = {
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 4096),
        }
        
        # OpenAI定价表 (2024年价格，需要定期更新)
        self.pricing = {
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        }
        
        # 使用tiktoken进行准确的token计数
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE:
            try:
                if "gpt-4" in model_name:
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
                else:
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")
            except Exception:
                logger.warning("Failed to load tiktoken encoder")
    
    def _generate_impl(self, prompt: str, **kwargs) -> str:
        """实际的OpenAI API调用"""
        params = {**self.default_params, **kwargs}
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **params
        )
        return response.choices[0].message.content.strip()
    
    def count_tokens(self, text: str) -> int:
        """准确的token计数"""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            return int(len(text.split()) * 1.3)
    
    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """估算API调用成本"""
        pricing = self.pricing.get(self.model_name, {"input": 0.002, "output": 0.002})
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        return input_cost + output_cost


class HuggingFaceLLM(BaseLLM):
    """Enhanced HuggingFace model interface with optimized memory management"""
    
    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.device = device
        
        logger.info(f"Loading HuggingFace model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 处理tokenizer的pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 内存优化的模型加载
        load_kwargs = {}
        if device == "cuda" and torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        if device != "auto":
            self.model.to(device)
        
        self.default_params = {
            "max_length": kwargs.get("max_tokens", 2048),
            "temperature": kwargs.get("temperature", 0.1),
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
    
    def _generate_impl(self, prompt: str, **kwargs) -> str:
        """实际的HuggingFace模型生成"""
        params = {**self.default_params, **kwargs}
        
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                **params
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # 移除输入prompt，只返回生成的部分
        response = generated_text[len(prompt):].strip()
        return response
    
    def count_tokens(self, text: str) -> int:
        """精确的token计数"""
        return len(self.tokenizer.encode(text))
    
    def _estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """本地模型无API成本"""
        return 0.0


# ============ Enhanced Unified Interface ============

class LLMInterface:
    """Enhanced unified LLM interface management class with advanced features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.llm = self._initialize_llm()
        
        # 并发控制
        self.max_concurrent = config.get("max_concurrent", 5)
        self.semaphore = threading.Semaphore(self.max_concurrent)
        
    def _initialize_llm(self) -> BaseLLM:
        """Initialize LLM based on configuration"""
        provider = self.config.get("provider", "openai").lower()
        model_name = self.config.get("model_name", "gpt-3.5-turbo")
        
        # 传递增强功能的配置
        enhanced_kwargs = {
            "temperature": self.config.get("temperature", 0.1),
            "max_tokens": self.config.get("max_tokens", 4096),
            "enable_cache": self.config.get("enable_cache", True),
            "cache_size": self.config.get("cache_size", 1000),
            "cache_ttl": self.config.get("cache_ttl", 3600),
            "rate_limit": self.config.get("rate_limit", 60),
            "failure_threshold": self.config.get("failure_threshold", 5),
            "recovery_timeout": self.config.get("recovery_timeout", 60.0)
        }
        
        if provider in ["openai", "deepseek"]:
            base_url = None
            api_key = None

            if provider == "deepseek":
                base_url = "https://api.deepseek.com"
                api_key = self.config.get("api_key") or os.getenv("DEEPSEEK_API_KEY")
                if model_name == "gpt-3.5-turbo":
                    model_name = "deepseek-chat"
                return DeepSeekLLM(
                    model_name=model_name,
                    api_key=api_key,
                    **enhanced_kwargs
                )
            else:  # openai
                api_key = self.config.get("api_key") or os.getenv("OPENAI_API_KEY")
                return OpenAILLM(
                    model_name=model_name,
                    api_key=api_key,
                    base_url=base_url,
                    **enhanced_kwargs
                )
        elif provider == "huggingface":
            enhanced_kwargs["device"] = self.config.get("device", "cuda")
            return HuggingFaceLLM(
                model_name=model_name,
                **enhanced_kwargs
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Enhanced generation with concurrency control"""
        with self.semaphore:
            return self.llm.generate(prompt, **kwargs)
    
    def generate_with_retry(self, prompt: str, max_retries: int = 3, 
                          backoff_factor: float = 2.0, **kwargs) -> str:
        """Enhanced retry with exponential backoff"""
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return self.generate(prompt, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    sleep_time = backoff_factor ** attempt
                    logger.warning(f"Attempt {attempt + 1} failed: {e}, retrying in {sleep_time}s")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"All {max_retries} attempts failed")
        
        raise last_exception
    
    def count_tokens(self, text: str) -> int:
        """Count tokens"""
        return self.llm.count_tokens(text)
    
    def batch_generate(self, prompts: List[str], max_workers: Optional[int] = None, **kwargs) -> List[str]:
        """Enhanced batch generation with concurrent processing"""
        if not prompts:
            return []
            
        max_workers = max_workers or min(len(prompts), self.max_concurrent)
        results = [""] * len(prompts)  # 预分配结果列表
        
        def generate_one(index_prompt_pair):
            index, prompt = index_prompt_pair
            try:
                return index, self.generate(prompt, **kwargs)
            except Exception as e:
                logger.error(f"Batch generation failed for prompt {index}: {e}")
                return index, ""
        
        # 使用ThreadPoolExecutor进行并发处理
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_index = {
                executor.submit(generate_one, (i, prompt)): i 
                for i, prompt in enumerate(prompts)
            }
            
            # 收集结果
            for future in as_completed(future_to_index):
                try:
                    index, result = future.result()
                    results[index] = result
                except Exception as e:
                    index = future_to_index[future]
                    logger.error(f"Future failed for index {index}: {e}")
                    results[index] = ""
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """获取LLM统计信息"""
        stats = self.llm.get_stats()
        stats["provider"] = self.config.get("provider")
        stats["model_name"] = self.config.get("model_name")
        stats["max_concurrent"] = self.max_concurrent
        return stats
    
    def reset_stats(self):
        """重置统计信息"""
        self.llm.stats = APICallStats()
        self.llm.circuit_breaker = CircuitBreaker(
            failure_threshold=self.config.get("failure_threshold", 5),
            recovery_timeout=self.config.get("recovery_timeout", 60.0)
        )
    
    def export_stats(self, filepath: str):
        """导出统计信息到文件"""
        stats = self.get_stats()
        stats["export_time"] = time.time()
        
        output_path = Path(filepath)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info(f"LLM统计信息已导出到: {filepath}")
    
    def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        try:
            test_prompt = "Hello, this is a test."
            start_time = time.time()
            response = self.generate(test_prompt, max_tokens=10)
            elapsed = time.time() - start_time
            
            return {
                "status": "healthy",
                "response_time": elapsed,
                "response_length": len(response),
                "circuit_breaker_state": self.llm.circuit_breaker.state
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "circuit_breaker_state": self.llm.circuit_breaker.state
            }
