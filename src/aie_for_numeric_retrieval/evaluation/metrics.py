"""
Evaluation Metrics Module (Enhanced)

Implement various evaluation metrics for measuring information extraction performance

Enhancements:
- RETA (Relative Error Tolerance Accuracy) metrics for financial data
- Confidence-weighted metrics and calibration analysis
- Multi-threshold evaluation with statistical significance tests
- Stratified evaluation by data types, companies, years
- Cached computation for expensive metrics
- Bootstrap confidence intervals and statistical tests
- Advanced financial-specific metrics (MAPE, SMAPE, directional accuracy)
"""

import re
import math
import logging
import hashlib
import json
from typing import List, Dict, Any, Union, Optional, Tuple, Set
from collections import Counter, defaultdict
from functools import lru_cache
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class EvaluationMetrics:
    """Enhanced evaluation metrics calculation class"""
    
    # ============ Core Metrics ============
    
    @staticmethod
    def exact_match(predicted: Any, target: Any) -> float:
        """exact match"""
        if predicted is None and target is None:
            return 1.0
        if predicted is None or target is None:
            return 0.0
        
        # 数值比较
        if isinstance(predicted, (int, float)) and isinstance(target, (int, float)):
            return 1.0 if abs(predicted - target) < 1e-6 else 0.0
        
        # characters串比较
        return 1.0 if str(predicted).strip() == str(target).strip() else 0.0
    
    @staticmethod
    def numerical_accuracy(predicted: Union[int, float, None], target: Union[int, float, None], 
                         tolerance: float = 0.01) -> float:
        """Numerical accuracy（相对误差容忍）"""
        if predicted is None and target is None:
            return 1.0
        if predicted is None or target is None:
            return 0.0
        
        try:
            pred_val = float(predicted)
            target_val = float(target)
            
            if target_val == 0:
                return 1.0 if abs(pred_val) < tolerance else 0.0
            
            relative_error = abs(pred_val - target_val) / abs(target_val)
            return 1.0 if relative_error <= tolerance else 0.0
            
        except (ValueError, TypeError):
            return 0.0
    
    @staticmethod
    def f1_score(predicted: str, target: str) -> float:
        """F1 score（基于token）"""
        if not predicted and not target:
            return 1.0
        if not predicted or not target:
            return 0.0
        
        # 分词
        pred_tokens = set(str(predicted).lower().split())
        target_tokens = set(str(target).lower().split())
        
        if not pred_tokens and not target_tokens:
            return 1.0
        if not pred_tokens or not target_tokens:
            return 0.0
        
        intersection = pred_tokens.intersection(target_tokens)
        
        precision = len(intersection) / len(pred_tokens)
        recall = len(intersection) / len(target_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def precision(predicted: str, target: str) -> float:
        """Precision"""
        if not predicted:
            return 1.0 if not target else 0.0
        
        pred_tokens = set(str(predicted).lower().split())
        target_tokens = set(str(target).lower().split())
        
        if not pred_tokens:
            return 0.0
        
        intersection = pred_tokens.intersection(target_tokens)
        return len(intersection) / len(pred_tokens)
    
    @staticmethod
    def recall(predicted: str, target: str) -> float:
        """Recall"""
        if not target:
            return 1.0
        
        pred_tokens = set(str(predicted).lower().split())
        target_tokens = set(str(target).lower().split())
        
        if not target_tokens:
            return 1.0
        
        intersection = pred_tokens.intersection(target_tokens)
        return len(intersection) / len(target_tokens)
    
    @staticmethod
    def rouge_l(predicted: str, target: str) -> float:
        """ROUGE-L分数（最长公共子序列）"""
        if not predicted and not target:
            return 1.0
        if not predicted or not target:
            return 0.0
        
        pred_tokens = str(predicted).lower().split()
        target_tokens = str(target).lower().split()
        
        # 计算最长公共子序列length
        def lcs_length(seq1, seq2):
            m, n = len(seq1), len(seq2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if seq1[i-1] == seq2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])
            
            return dp[m][n]
        
        lcs_len = lcs_length(pred_tokens, target_tokens)
        
        if len(pred_tokens) == 0 and len(target_tokens) == 0:
            return 1.0
        if len(pred_tokens) == 0 or len(target_tokens) == 0:
            return 0.0
        
        precision = lcs_len / len(pred_tokens)
        recall = lcs_len / len(target_tokens)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * precision * recall / (precision + recall)
    
    @staticmethod
    def bleu_score(predicted: str, target: str, n: int = 4) -> float:
        """BLEU分数"""
        if not predicted and not target:
            return 1.0
        if not predicted or not target:
            return 0.0
        
        pred_tokens = str(predicted).lower().split()
        target_tokens = str(target).lower().split()
        
        if not pred_tokens or not target_tokens:
            return 0.0
        
        # 计算n-gramPrecision
        def get_ngrams(tokens, n):
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        
        precisions = []
        for i in range(1, min(n + 1, len(pred_tokens) + 1)):
            pred_ngrams = Counter(get_ngrams(pred_tokens, i))
            target_ngrams = Counter(get_ngrams(target_tokens, i))
            
            overlap = pred_ngrams & target_ngrams
            overlap_count = sum(overlap.values())
            total_pred_count = sum(pred_ngrams.values())
            
            if total_pred_count == 0:
                precision = 0.0
            else:
                precision = overlap_count / total_pred_count
            
            precisions.append(precision)
        
        if not precisions or all(p == 0 for p in precisions):
            return 0.0
        
        # 几何平均
        geometric_mean = math.exp(sum(math.log(p) if p > 0 else float('-inf') for p in precisions) / len(precisions))
        
        # 简化的length惩罚
        bp = 1.0 if len(pred_tokens) >= len(target_tokens) else math.exp(1 - len(target_tokens) / len(pred_tokens))
        
        return bp * geometric_mean
    
    # ============ RETA and Financial Metrics ============
    
    @staticmethod
    def reta_accuracy(predicted_values: List[Union[int, float, None]], 
                     target_values: List[Union[int, float, None]], 
                     tolerance: float = 0.03) -> float:
        """
        RETA (Relative Error Tolerance Accuracy) - 金融数据标准评估指标
        
        Args:
            predicted_values: 预测值列表
            target_values: 真实值列表  
            tolerance: 相对误差容忍度 (默认3%)
            
        Returns:
            RETA准确率 (0-1)
        """
        if not predicted_values or not target_values:
            return 0.0
            
        correct = 0
        valid_pairs = 0
        
        for pred, target in zip(predicted_values, target_values):
            if pred is None or target is None:
                continue
                
            try:
                pred_val = float(pred)
                target_val = float(target)
                valid_pairs += 1
                
                if target_val == 0:
                    # 对于零值，使用绝对误差
                    if abs(pred_val) <= tolerance:
                        correct += 1
                else:
                    # 相对误差
                    relative_error = abs(pred_val - target_val) / abs(target_val)
                    if relative_error <= tolerance:
                        correct += 1
                        
            except (ValueError, TypeError, ZeroDivisionError):
                continue
        
        return correct / valid_pairs if valid_pairs > 0 else 0.0
    
    @staticmethod
    def multi_threshold_reta(predicted_values: List[Union[int, float, None]], 
                            target_values: List[Union[int, float, None]], 
                            thresholds: List[float] = None) -> Dict[str, float]:
        """多阈值RETA评估"""
        if thresholds is None:
            thresholds = [0.01, 0.03, 0.05, 0.10, 0.15, 0.20]
            
        results = {}
        for threshold in thresholds:
            reta = EvaluationMetrics.reta_accuracy(predicted_values, target_values, threshold)
            results[f"RETA@{int(threshold*100)}%"] = reta
            
        return results
    
    @staticmethod
    def mape(predicted_values: List[Union[int, float, None]], 
             target_values: List[Union[int, float, None]]) -> float:
        """Mean Absolute Percentage Error"""
        valid_pairs = []
        for pred, target in zip(predicted_values, target_values):
            if pred is not None and target is not None:
                try:
                    pred_val = float(pred)
                    target_val = float(target)
                    if target_val != 0:  # 避免除零
                        valid_pairs.append((pred_val, target_val))
                except (ValueError, TypeError):
                    continue
        
        if not valid_pairs:
            return float('inf')
        
        errors = [abs(pred - target) / abs(target) for pred, target in valid_pairs]
        return sum(errors) / len(errors) * 100  # 返回百分比
    
    @staticmethod
    def smape(predicted_values: List[Union[int, float, None]], 
              target_values: List[Union[int, float, None]]) -> float:
        """Symmetric Mean Absolute Percentage Error"""
        valid_pairs = []
        for pred, target in zip(predicted_values, target_values):
            if pred is not None and target is not None:
                try:
                    pred_val = float(pred)
                    target_val = float(target)
                    valid_pairs.append((pred_val, target_val))
                except (ValueError, TypeError):
                    continue
        
        if not valid_pairs:
            return float('inf')
        
        errors = []
        for pred, target in valid_pairs:
            denominator = (abs(pred) + abs(target)) / 2
            if denominator != 0:
                errors.append(abs(pred - target) / denominator)
        
        return sum(errors) / len(errors) * 100 if errors else float('inf')
    
    @staticmethod
    def directional_accuracy(predicted_values: List[Union[int, float, None]], 
                           target_values: List[Union[int, float, None]], 
                           baseline_values: List[Union[int, float, None]] = None) -> float:
        """方向准确率（预测变化方向的准确性）"""
        if baseline_values is None:
            # 如果没有基准值，使用前一个值作为基准
            baseline_values = [None] + target_values[:-1]
        
        correct = 0
        valid_pairs = 0
        
        for pred, target, baseline in zip(predicted_values, target_values, baseline_values):
            if any(v is None for v in [pred, target, baseline]):
                continue
                
            try:
                pred_val = float(pred)
                target_val = float(target)
                baseline_val = float(baseline)
                
                pred_direction = 1 if pred_val > baseline_val else (-1 if pred_val < baseline_val else 0)
                target_direction = 1 if target_val > baseline_val else (-1 if target_val < baseline_val else 0)
                
                if pred_direction == target_direction:
                    correct += 1
                valid_pairs += 1
                
            except (ValueError, TypeError):
                continue
        
        return correct / valid_pairs if valid_pairs > 0 else 0.0
    
    @staticmethod
    def mean_absolute_error(predicted_values: List[Union[int, float, None]], 
                          target_values: List[Union[int, float, None]]) -> float:
        """Mean absolute error"""
        valid_pairs = []
        for pred, target in zip(predicted_values, target_values):
            if pred is not None and target is not None:
                try:
                    valid_pairs.append((float(pred), float(target)))
                except (ValueError, TypeError):
                    continue
        
        if not valid_pairs:
            return float('inf')
        
        errors = [abs(pred - target) for pred, target in valid_pairs]
        return sum(errors) / len(errors)
    
    @staticmethod
    def root_mean_square_error(predicted_values: List[Union[int, float, None]], 
                             target_values: List[Union[int, float, None]]) -> float:
        """Root mean square error"""
        valid_pairs = []
        for pred, target in zip(predicted_values, target_values):
            if pred is not None and target is not None:
                try:
                    valid_pairs.append((float(pred), float(target)))
                except (ValueError, TypeError):
                    continue
        
        if not valid_pairs:
            return float('inf')
        
        squared_errors = [(pred - target) ** 2 for pred, target in valid_pairs]
        return math.sqrt(sum(squared_errors) / len(squared_errors))
    
    # ============ Confidence-Weighted Metrics ============
    
    @staticmethod
    def confidence_weighted_accuracy(predicted_values: List[Any], 
                                   target_values: List[Any], 
                                   confidence_scores: List[float],
                                   threshold: float = 0.5) -> Dict[str, float]:
        """置信度加权准确率"""
        if not confidence_scores:
            confidence_scores = [1.0] * len(predicted_values)
            
        total_weight = 0.0
        weighted_correct = 0.0
        high_conf_correct = 0
        high_conf_total = 0
        
        for pred, target, conf in zip(predicted_values, target_values, confidence_scores):
            if pred is None or target is None:
                continue
                
            weight = max(0.0, min(1.0, float(conf)))
            total_weight += weight
            
            is_correct = EvaluationMetrics.exact_match(pred, target)
            weighted_correct += is_correct * weight
            
            if conf >= threshold:
                high_conf_total += 1
                high_conf_correct += is_correct
        
        results = {
            "weighted_accuracy": weighted_correct / total_weight if total_weight > 0 else 0.0,
            "high_confidence_accuracy": high_conf_correct / high_conf_total if high_conf_total > 0 else 0.0,
            "high_confidence_ratio": high_conf_total / len(predicted_values) if predicted_values else 0.0
        }
        
        return results
    
    @staticmethod
    def calibration_metrics(predicted_values: List[Any], 
                          target_values: List[Any], 
                          confidence_scores: List[float],
                          n_bins: int = 10) -> Dict[str, float]:
        """置信度校准指标"""
        if not confidence_scores or len(confidence_scores) != len(predicted_values):
            return {"ece": 0.0, "mce": 0.0, "reliability": 0.0}
            
        # 创建置信度区间
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0  # Expected Calibration Error
        mce = 0.0  # Maximum Calibration Error
        total_samples = len(predicted_values)
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # 找到在此区间的样本
            in_bin = []
            for i, conf in enumerate(confidence_scores):
                if bin_lower <= conf < bin_upper or (bin_upper == 1.0 and conf == 1.0):
                    in_bin.append(i)
            
            if not in_bin:
                continue
                
            # 计算该区间的准确率和平均置信度
            bin_accuracy = 0.0
            bin_confidence = 0.0
            
            for i in in_bin:
                if predicted_values[i] is not None and target_values[i] is not None:
                    bin_accuracy += EvaluationMetrics.exact_match(predicted_values[i], target_values[i])
                bin_confidence += confidence_scores[i]
            
            bin_accuracy /= len(in_bin)
            bin_confidence /= len(in_bin)
            
            # 计算校准误差
            calibration_error = abs(bin_accuracy - bin_confidence)
            ece += (len(in_bin) / total_samples) * calibration_error
            mce = max(mce, calibration_error)
        
        return {
            "ece": ece,  # Expected Calibration Error
            "mce": mce,  # Maximum Calibration Error
            "reliability": 1.0 - ece  # 可靠性分数
        }
    
    # ============ Statistical Analysis ============
    
    @staticmethod
    def bootstrap_confidence_interval(predicted_values: List[Any], 
                                    target_values: List[Any], 
                                    metric_func: callable,
                                    n_bootstrap: int = 1000, 
                                    confidence_level: float = 0.95) -> Tuple[float, float, float]:
        """Bootstrap置信区间"""
        if len(predicted_values) != len(target_values):
            return 0.0, 0.0, 0.0
            
        # 原始指标值
        original_metric = metric_func(predicted_values, target_values)
        
        # Bootstrap重采样
        bootstrap_metrics = []
        n_samples = len(predicted_values)
        
        for _ in range(n_bootstrap):
            # 有放回抽样
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            boot_pred = [predicted_values[i] for i in indices]
            boot_target = [target_values[i] for i in indices]
            
            try:
                boot_metric = metric_func(boot_pred, boot_target)
                if not (math.isnan(boot_metric) or math.isinf(boot_metric)):
                    bootstrap_metrics.append(boot_metric)
            except:
                continue
        
        if not bootstrap_metrics:
            return original_metric, original_metric, original_metric
            
        # 计算置信区间
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        ci_lower = np.percentile(bootstrap_metrics, lower_percentile)
        ci_upper = np.percentile(bootstrap_metrics, upper_percentile)
        
        return original_metric, ci_lower, ci_upper
    
    @staticmethod
    def significance_test(predicted_values_1: List[Any], 
                         target_values_1: List[Any],
                         predicted_values_2: List[Any], 
                         target_values_2: List[Any],
                         metric_func: callable) -> Dict[str, float]:
        """统计显著性检验（比较两个模型）"""
        # 计算每个样本的指标值
        scores_1 = []
        scores_2 = []
        
        for pred, target in zip(predicted_values_1, target_values_1):
            if pred is not None and target is not None:
                score = metric_func([pred], [target])
                scores_1.append(score)
        
        for pred, target in zip(predicted_values_2, target_values_2):
            if pred is not None and target is not None:
                score = metric_func([pred], [target])
                scores_2.append(score)
        
        if not scores_1 or not scores_2:
            return {"p_value": 1.0, "statistic": 0.0, "significant": False}
        
        # 进行t检验
        try:
            statistic, p_value = stats.ttest_ind(scores_1, scores_2)
            return {
                "p_value": p_value,
                "statistic": statistic,
                "significant": p_value < 0.05,
                "mean_1": np.mean(scores_1),
                "mean_2": np.mean(scores_2),
                "std_1": np.std(scores_1),
                "std_2": np.std(scores_2)
            }
        except:
            return {"p_value": 1.0, "statistic": 0.0, "significant": False}
    
    # ============ Stratified Evaluation ============
    
    @staticmethod
    def stratified_evaluation(extraction_results: List[Dict[str, Any]], 
                            ground_truth: List[Dict[str, Any]], 
                            stratify_by: str = "target_type") -> Dict[str, Dict[str, float]]:
        """分层评估（按数据类型、公司、年份等分层）"""
        # 按分层字段分组
        groups = defaultdict(lambda: {"predicted": [], "target": [], "confidence": []})
        
        for result, truth in zip(extraction_results, ground_truth):
            # 确定分层标签
            if stratify_by == "target_type":
                group_key = result.get("data_type", truth.get("data_type", "unknown"))
            elif stratify_by == "company":
                group_key = result.get("company", truth.get("company", "unknown"))
            elif stratify_by == "year":
                group_key = str(result.get("year", truth.get("year", "unknown")))
            elif stratify_by == "confidence_level":
                conf = result.get("confidence", 0.0)
                if conf >= 0.8:
                    group_key = "high"
                elif conf >= 0.5:
                    group_key = "medium"
                else:
                    group_key = "low"
            else:
                group_key = str(result.get(stratify_by, truth.get(stratify_by, "unknown")))
            
            groups[group_key]["predicted"].append(result.get("value"))
            groups[group_key]["target"].append(truth.get("value"))
            groups[group_key]["confidence"].append(result.get("confidence", 0.0))
        
        # 为每个分层计算指标
        stratified_results = {}
        
        for group_key, data in groups.items():
            if not data["predicted"]:
                continue
                
            # 选择合适的指标
            sample_target = next((t for t in data["target"] if t is not None), None)
            if sample_target is not None and isinstance(sample_target, (int, float)):
                # 数值类型指标
                metrics = {
                    "count": len(data["predicted"]),
                    "exact_match": EvaluationMetrics.exact_match(data["predicted"][0], data["target"][0]) if data["predicted"] and data["target"] else 0.0,
                    **EvaluationMetrics.multi_threshold_reta(data["predicted"], data["target"]),
                    "mae": EvaluationMetrics.mean_absolute_error(data["predicted"], data["target"]),
                    "rmse": EvaluationMetrics.root_mean_square_error(data["predicted"], data["target"]),
                    "mape": EvaluationMetrics.mape(data["predicted"], data["target"]),
                    "avg_confidence": np.mean(data["confidence"]) if data["confidence"] else 0.0
                }
            else:
                # 文本类型指标
                exact_matches = [EvaluationMetrics.exact_match(p, t) for p, t in zip(data["predicted"], data["target"])]
                f1_scores = [EvaluationMetrics.f1_score(str(p) if p else "", str(t) if t else "") 
                           for p, t in zip(data["predicted"], data["target"])]
                
                metrics = {
                    "count": len(data["predicted"]),
                    "exact_match": np.mean(exact_matches) if exact_matches else 0.0,
                    "f1_score": np.mean(f1_scores) if f1_scores else 0.0,
                    "avg_confidence": np.mean(data["confidence"]) if data["confidence"] else 0.0
                }
            
            stratified_results[group_key] = metrics
        
        return stratified_results
    
    @classmethod
    def compute_all_metrics(cls, predicted_values: List[Any], target_values: List[Any], 
                          metric_types: Optional[List[str]] = None) -> Dict[str, float]:
        """Compute all metrics"""
        if metric_types is None:
            metric_types = ["exact_match", "f1_score", "precision", "recall", "rouge_l"]
        
        results = {}
        
        # Compute metrics pairwise
        pairwise_results = {metric: [] for metric in metric_types}
        
        for pred, target in zip(predicted_values, target_values):
            if "exact_match" in metric_types:
                pairwise_results["exact_match"].append(cls.exact_match(pred, target))
            
            if "numerical_accuracy" in metric_types:
                pairwise_results["numerical_accuracy"].append(cls.numerical_accuracy(pred, target))
            
            # 文本相关指标
            pred_str = str(pred) if pred is not None else ""
            target_str = str(target) if target is not None else ""
            
            if "f1_score" in metric_types:
                pairwise_results["f1_score"].append(cls.f1_score(pred_str, target_str))
            
            if "precision" in metric_types:
                pairwise_results["precision"].append(cls.precision(pred_str, target_str))
            
            if "recall" in metric_types:
                pairwise_results["recall"].append(cls.recall(pred_str, target_str))
            
            if "rouge_l" in metric_types:
                pairwise_results["rouge_l"].append(cls.rouge_l(pred_str, target_str))
            
            if "bleu_score" in metric_types:
                pairwise_results["bleu_score"].append(cls.bleu_score(pred_str, target_str))
        
        # Calculate average
        for metric, scores in pairwise_results.items():
            if scores:
                results[metric] = sum(scores) / len(scores)
            else:
                results[metric] = 0.0
        
        # Global metrics for numerical values
        if "mae" in metric_types:
            results["mae"] = cls.mean_absolute_error(predicted_values, target_values)
        
        if "rmse" in metric_types:
            results["rmse"] = cls.root_mean_square_error(predicted_values, target_values)
        
        # 新增RETA和金融指标
        if any(m in metric_types for m in ["reta", "multi_reta", "mape", "smape", "directional_accuracy"]):
            if "reta" in metric_types:
                results["reta"] = cls.reta_accuracy(predicted_values, target_values)
            if "multi_reta" in metric_types:
                results.update(cls.multi_threshold_reta(predicted_values, target_values))
            if "mape" in metric_types:
                results["mape"] = cls.mape(predicted_values, target_values)
            if "smape" in metric_types:
                results["smape"] = cls.smape(predicted_values, target_values)
            if "directional_accuracy" in metric_types:
                results["directional_accuracy"] = cls.directional_accuracy(predicted_values, target_values)
        
        return results
    
    @classmethod
    def comprehensive_evaluation(cls, extraction_results: List[Dict[str, Any]], 
                               ground_truth: List[Dict[str, Any]], 
                               include_confidence: bool = True,
                               include_bootstrap: bool = False,
                               include_stratified: bool = True) -> Dict[str, Any]:
        """全面评估（包含所有增强功能）"""
        
        # 基础评估
        basic_results = cls.evaluate_extraction_results(extraction_results, ground_truth)
        
        # 提取数据
        predicted_values = [r.get("value") for r in extraction_results]
        target_values = [t.get("value") for t in ground_truth]
        confidence_scores = [r.get("confidence", 0.0) for r in extraction_results]
        
        comprehensive_results = {
            "basic_metrics": basic_results,
            "enhanced_metrics": {}
        }
        
        # 置信度相关指标
        if include_confidence and confidence_scores:
            confidence_metrics = cls.confidence_weighted_accuracy(
                predicted_values, target_values, confidence_scores
            )
            calibration_metrics = cls.calibration_metrics(
                predicted_values, target_values, confidence_scores
            )
            comprehensive_results["enhanced_metrics"]["confidence"] = {
                **confidence_metrics,
                **calibration_metrics
            }
        
        # Bootstrap置信区间
        if include_bootstrap and len(predicted_values) > 10:
            try:
                reta_ci = cls.bootstrap_confidence_interval(
                    predicted_values, target_values, 
                    lambda p, t: cls.reta_accuracy(p, t)
                )
                comprehensive_results["enhanced_metrics"]["bootstrap"] = {
                    "reta_ci_mean": reta_ci[0],
                    "reta_ci_lower": reta_ci[1],
                    "reta_ci_upper": reta_ci[2]
                }
            except Exception as e:
                logger.warning(f"Bootstrap分析失败: {e}")
        
        # 分层评估
        if include_stratified:
            stratified_results = {}
            for stratify_field in ["target_type", "confidence_level"]:
                try:
                    strat_results = cls.stratified_evaluation(
                        extraction_results, ground_truth, stratify_field
                    )
                    if strat_results:
                        stratified_results[stratify_field] = strat_results
                except Exception as e:
                    logger.warning(f"分层评估失败 ({stratify_field}): {e}")
            
            if stratified_results:
                comprehensive_results["enhanced_metrics"]["stratified"] = stratified_results
        
        # RETA多阈值分析
        numerical_results = [(p, t) for p, t in zip(predicted_values, target_values) 
                           if p is not None and t is not None and 
                           isinstance(p, (int, float)) and isinstance(t, (int, float))]
        
        if numerical_results:
            num_pred, num_target = zip(*numerical_results)
            multi_reta = cls.multi_threshold_reta(list(num_pred), list(num_target))
            financial_metrics = {
                "mape": cls.mape(list(num_pred), list(num_target)),
                "smape": cls.smape(list(num_pred), list(num_target)),
                "directional_accuracy": cls.directional_accuracy(list(num_pred), list(num_target))
            }
            comprehensive_results["enhanced_metrics"]["financial"] = {
                **multi_reta,
                **financial_metrics
            }
        
        return comprehensive_results
    
    @classmethod
    def evaluate_extraction_results(cls, extraction_results: List[Dict[str, Any]], 
                                  ground_truth: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估information extraction结果"""
        # Organize data by target name
        target_results = {}
        
        for result, truth in zip(extraction_results, ground_truth):
            target_name = result.get("target_name", "unknown")
            
            if target_name not in target_results:
                target_results[target_name] = {
                    "predicted": [],
                    "target": [],
                    "confidence": []
                }
            
            target_results[target_name]["predicted"].append(result.get("value"))
            target_results[target_name]["target"].append(truth.get("value"))
            target_results[target_name]["confidence"].append(result.get("confidence", 0.0))
        
        # Calculate metrics for each target
        detailed_results = {}
        overall_metrics = {"exact_match": [], "f1_score": [], "precision": [], "recall": []}
        
        for target_name, data in target_results.items():
            # Select appropriate metrics based on data type
            target_type = ground_truth[0].get("data_type", "text") if ground_truth else "text"
            
            if target_type in ["number", "float", "integer"]:
                metrics = cls.compute_all_metrics(
                    data["predicted"], 
                    data["target"],
                    ["exact_match", "numerical_accuracy", "mae", "rmse"]
                )
            else:
                metrics = cls.compute_all_metrics(
                    data["predicted"], 
                    data["target"],
                    ["exact_match", "f1_score", "precision", "recall", "rouge_l"]
                )
            
            # Add confidence-related statistics
            confidences = data["confidence"]
            metrics["avg_confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
            metrics["confidence_std"] = np.std(confidences) if confidences else 0.0
            
            detailed_results[target_name] = metrics
            
            # Accumulate overall metrics
            for metric in ["exact_match", "f1_score", "precision", "recall"]:
                if metric in metrics:
                    overall_metrics[metric].append(metrics[metric])
        
        # Calculate overall metrics
        overall_results = {}
        for metric, values in overall_metrics.items():
            if values:
                overall_results[f"overall_{metric}"] = sum(values) / len(values)
            else:
                overall_results[f"overall_{metric}"] = 0.0
        
        return {
            "overall": overall_results,
            "by_target": detailed_results,
            "summary": {
                "total_targets": len(target_results),
                "total_samples": len(extraction_results),
                "average_confidence": np.mean([r.get("confidence", 0.0) for r in extraction_results])
            }
        }
    
    @staticmethod
    def print_evaluation_report(evaluation_results: Dict[str, Any]):
        """Print evaluation report"""
        print("\\n" + "="*60)
        print("           information extraction评估报告")
        print("="*60)
        
        # Overall results
        print("\\n整体性能:")
        print("-" * 30)
        overall = evaluation_results.get("overall", {})
        for metric, value in overall.items():
            print(f"{metric:20}: {value:.4f}")
        
        # Summary information
        print("\\n摘要统计:")
        print("-" * 30)
        summary = evaluation_results.get("summary", {})
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key:20}: {value:.4f}")
            else:
                print(f"{key:20}: {value}")
        
        # Detailed results by target
        print("\\nDetailed results by target:")
        print("-" * 30)
        by_target = evaluation_results.get("by_target", {})
        for target_name, metrics in by_target.items():
            print(f"\\n{target_name}:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {metric:18}: {value:.4f}")
                else:
                    print(f"  {metric:18}: {value}")
        
        print("\\n" + "="*60)
    
    @staticmethod
    def export_detailed_results(evaluation_results: Dict[str, Any], 
                              output_path: str, 
                              format: str = "json") -> str:
        """导出详细评估结果"""
        from pathlib import Path
        import json
        import csv
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "json":
            with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2, default=str)
            return str(output_path.with_suffix('.json'))
        
        elif format.lower() == "csv":
            # 导出为CSV格式（扁平化）
            rows = []
            
            # 整体指标
            overall = evaluation_results.get("overall", {})
            for metric, value in overall.items():
                rows.append({"category": "overall", "metric": metric, "value": value})
            
            # 按目标分类的指标
            by_target = evaluation_results.get("by_target", {})
            for target_name, metrics in by_target.items():
                for metric, value in metrics.items():
                    rows.append({"category": f"target_{target_name}", "metric": metric, "value": value})
            
            # 增强指标
            enhanced = evaluation_results.get("enhanced_metrics", {})
            for category, metrics in enhanced.items():
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        rows.append({"category": f"enhanced_{category}", "metric": metric, "value": value})
            
            with open(output_path.with_suffix('.csv'), 'w', newline='', encoding='utf-8') as f:
                if rows:
                    writer = csv.DictWriter(f, fieldnames=["category", "metric", "value"])
                    writer.writeheader()
                    writer.writerows(rows)
            
            return str(output_path.with_suffix('.csv'))
        
        else:
            raise ValueError(f"不支持的格式: {format}")
    
    @staticmethod
    def generate_evaluation_report(evaluation_results: Dict[str, Any], 
                                 output_path: Optional[str] = None) -> str:
        """生成详细的评估报告"""
        report_lines = []
        
        # 标题
        report_lines.append("=" * 80)
        report_lines.append("           AIE框架信息提取评估报告")
        report_lines.append("=" * 80)
        
        # 整体性能
        overall = evaluation_results.get("overall", {})
        if overall:
            report_lines.append("\\n📊 整体性能指标:")
            report_lines.append("-" * 50)
            for metric, value in overall.items():
                if isinstance(value, float):
                    report_lines.append(f"  {metric:25}: {value:.4f}")
                else:
                    report_lines.append(f"  {metric:25}: {value}")
        
        # 增强指标
        enhanced = evaluation_results.get("enhanced_metrics", {})
        
        # RETA和金融指标
        financial = enhanced.get("financial", {})
        if financial:
            report_lines.append("\\n💰 金融专用指标:")
            report_lines.append("-" * 50)
            for metric, value in financial.items():
                if isinstance(value, float):
                    if "RETA" in metric:
                        report_lines.append(f"  {metric:25}: {value:.4f}")
                    elif metric in ["mape", "smape"]:
                        report_lines.append(f"  {metric:25}: {value:.2f}%")
                    else:
                        report_lines.append(f"  {metric:25}: {value:.4f}")
        
        # 置信度指标
        confidence = enhanced.get("confidence", {})
        if confidence:
            report_lines.append("\\n🎯 置信度分析:")
            report_lines.append("-" * 50)
            for metric, value in confidence.items():
                if isinstance(value, float):
                    report_lines.append(f"  {metric:25}: {value:.4f}")
        
        # 分层评估
        stratified = enhanced.get("stratified", {})
        if stratified:
            report_lines.append("\\n📈 分层评估结果:")
            report_lines.append("-" * 50)
            for strat_type, strat_results in stratified.items():
                report_lines.append(f"\\n  按 {strat_type} 分层:")
                for group, metrics in strat_results.items():
                    report_lines.append(f"    {group}:")
                    for metric, value in metrics.items():
                        if isinstance(value, float) and metric != "count":
                            report_lines.append(f"      {metric:20}: {value:.4f}")
                        else:
                            report_lines.append(f"      {metric:20}: {value}")
        
        # Bootstrap置信区间
        bootstrap = enhanced.get("bootstrap", {})
        if bootstrap:
            report_lines.append("\\n📊 统计置信区间:")
            report_lines.append("-" * 50)
            for metric, value in bootstrap.items():
                if isinstance(value, float):
                    report_lines.append(f"  {metric:25}: {value:.4f}")
        
        # 按目标详细结果
        by_target = evaluation_results.get("by_target", {})
        if by_target:
            report_lines.append("\\n🎯 按提取目标详细结果:")
            report_lines.append("-" * 50)
            for target_name, metrics in by_target.items():
                report_lines.append(f"\\n  目标: {target_name}")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        report_lines.append(f"    {metric:22}: {value:.4f}")
                    else:
                        report_lines.append(f"    {metric:22}: {value}")
        
        # 摘要统计
        summary = evaluation_results.get("summary", {})
        if summary:
            report_lines.append("\\n📋 摘要统计:")
            report_lines.append("-" * 50)
            for key, value in summary.items():
                if isinstance(value, float):
                    report_lines.append(f"  {key:25}: {value:.4f}")
                else:
                    report_lines.append(f"  {key:25}: {value}")
        
        report_lines.append("\\n" + "=" * 80)
        
        report_content = "\\n".join(report_lines)
        
        # 保存到文件
        if output_path:
            from pathlib import Path
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            logger.info(f"评估报告已保存到: {output_path}")
        
        return report_content
