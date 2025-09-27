import os
import json
import gzip
import requests
from argparse import ArgumentParser
from datasets import load_dataset
from datasets.exceptions import DatasetGenerationError
#####    python src/data/data_get.py --mode download     #####
BASE_DIR = "data/testing_data"

# 定义数据集及其目标目录
DATASETS = {
    # domain: {dataset_name: (hf_id, backup_hf_id_or_None)}
    "finance": {
        "docfinqa": ("kensho/DocFinQA", None),
        "tatqa": ("next-tat/TAT-QA", "ibm/TAT-QA"),  # 备选源
        "finqa": ("dreamerdeo/finqa", None),
        "convfinqa": ("ChilleD/ConvFinQA", None),
        "multihiertt": ("microsoft/MultiHiertt", None),
        "finer": ("nlpaueb/finer-139", None),
    },
    "government": {
        "govreport": ("launch/gov_report", "ccdv/govreport-summarization"),
        "qmsum": ("pszemraj/qmsum-cleaned", None),
        "billsum": ("FiscalNote/BillSum", None),
        "multinews": ("multi_news", None),
        "cnn_dailymail": ("cnn_dailymail", None),
    },
    "science_law": {
        "pubmedqa": ("bigbio/pubmed_qa", "pubmed_qa"),
        "cuad": ("theatticusproject/cuad", None),
        "scifact": ("scifact", None),
        "evidence_infer": ("evidence_infer_treatment", None),
        "legal_pile": ("pile-of-law/pile-of-law", None),
        "case_hold": ("casehold", None),
    },
    "general_qa": {
        "squad": ("squad", None),
        "squad_v2": ("squad_v2", None),
        "natural_questions": ("natural_questions", None),
        "ms_marco": ("ms_marco", None),
        "hotpotqa": ("hotpot_qa", None),
        "narrativeqa": ("narrativeqa", None),
    },
    "summarization": {
        "xsum": ("xsum", None),
        "reddit_tifu": ("reddit_tifu", None),
        "booksum": ("kmfoda/booksum", None),
        "arxiv": ("scientific_papers", None),
        "pubmed": ("scientific_papers", None),
    },
}

def save_split(dataset, split_name, out_path, domain, dataset_name, compress=False):
    """保存单个 split 为 JSONL（可选压缩）"""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    opener = gzip.open if compress else open
    mode = "wt" if compress else "w"
    
    with opener(out_path, mode, encoding="utf-8") as f:
        for i, item in enumerate(dataset):
            record = build_record_safe(item, dataset_name, split_name, i, domain)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

def build_record_safe(item, dataset_name, split_name, idx, domain):
    """按照 AIE Pipeline 预期格式构造记录，并对不可序列化对象做安全处理。

    输出字段：
    - document_id: 唯一样本ID（字符串）
    - document_text: 文本内容（字符串）
    - query: 用户问题/查询（字符串，可为空）
    - metadata: 其他元信息（字典），包含 domain/dataset/split/answers/原始剩余字段
    """
    def to_json_safe(obj):
        # 基础类型直接返回
        if obj is None or isinstance(obj, (bool, int, float, str)):
            return obj
        # bytes → 尝试utf-8
        if isinstance(obj, (bytes, bytearray)):
            try:
                return obj.decode("utf-8", errors="ignore")
            except Exception:
                return None
        # 列表 / 元组
        if isinstance(obj, (list, tuple)):
            return [to_json_safe(x) for x in obj]
        # 字典
        if isinstance(obj, dict):
            return {str(k): to_json_safe(v) for k, v in obj.items()}
        # 其他对象 → 字符串表示，防止PDF等对象报错
        try:
            s = str(obj)
            # 限制字符串过长
            if len(s) > 200000:
                return s[:200000] + "…"
            return s
        except Exception:
            return None

    # 常见字段抽取（扩展候选，兼容不同数据集）
    doc_candidates = [
        item.get("context"), item.get("document"), item.get("text"),
        item.get("input"), item.get("passage"), item.get("dialogue"),
        item.get("transcript"), item.get("article"), item.get("content"),
    ]
    document = next((d for d in doc_candidates if isinstance(d, (str, list, tuple)) and d), "")
    if isinstance(document, (list, tuple)):
        document = "\n".join([str(x) for x in document])
    question = item.get("question") or item.get("query") or item.get("instruction") or None
    answers = item.get("answers") or item.get("answer") or []

    # 元数据：排除已提取字段，随后与 domain/dataset/split/answers 合并
    metadata_raw = {k: v for k, v in item.items() if k not in ["context", "document", "text", "question", "query", "answers", "answer", "input", "passage", "dialogue", "transcript", "article", "content"]}

    aie_record = {
        "document_id": f"{dataset_name}_{split_name}_{idx}",
        "document_text": to_json_safe(document) or "",
        "query": (to_json_safe(question) or ""),
        "metadata": to_json_safe({
            "domain": domain,
            "dataset": dataset_name,
            "split": split_name,
            "answers": to_json_safe(answers),
            **metadata_raw,
        }),
    }
    return aie_record

def load_raw_json_dataset(dataset_id: str, file_urls: dict):
    """直接从原始JSON文件加载数据集，绕过HuggingFace的Arrow转换"""
    import io

    class RawJSONDataset:
        def __init__(self, splits_data):
            self.splits_data = splits_data
        def keys(self):
            return self.splits_data.keys()
        def __getitem__(self, split):
            return self.splits_data[split]

    splits_data = {}
    for split, url in file_urls.items():
        try:
            print(f"  📥 Downloading {split} from raw JSON...")
            if url.startswith("hf://"):
                # 转换HF URL为直接下载链接
                url = url.replace("hf://datasets/", "https://huggingface.co/datasets/").replace("@", "/resolve/") + "?download=true"

            # ✅ 流式 + 超时
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                buf = io.StringIO()
                for chunk in r.iter_content(chunk_size=1 << 16):
                    if chunk:
                        buf.write(chunk.decode("utf-8", errors="ignore"))
                content = buf.getvalue()

            data = []
            # JSONL 情况
            if "\n" in content and not content.strip().startswith('['):
                for line in content.strip().splitlines():
                    line = line.strip()
                    if line:
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            else:
                # 标准 JSON（list 或 dict）
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list):
                        data = parsed
                    elif isinstance(parsed, dict):
                        data = [parsed]
                except json.JSONDecodeError as e:
                    print(f"  ❌ JSON decode error for {split}: {e}")
                    continue

            splits_data[split] = data
            print(f"  ✅ Loaded {len(data)} samples for {split}")

        except Exception as e:
            print(f"  ❌ Error loading {split}: {e}")
            continue

    return RawJSONDataset(splits_data) if splits_data else None

def robust_load_dataset(hf_id: str, backup_id: str = None):
    """尽量健壮地加载HF数据集：
    - TAT-QA：强制走原始 JSON，彻底绕过 Arrow
    - PubMedQA：按常见 config 轮询加载（bigbio 与原版都试）
    - 其余：主源 → 备选源 → streaming/ignore_verifications/no_checks 等兜底
    """

    # === 特判 1：TAT-QA 直接拉原始 JSON，避免 ArrowInvalid ===
    if hf_id in ["next-tat/TAT-QA", "ibm/TAT-QA"]:
        print(f"  🔄 Forcing direct JSON loading for {hf_id}")
        file_urls = {
            "train": "https://huggingface.co/datasets/next-tat/TAT-QA/resolve/main/tatqa_dataset_train.json",
            "validation": "https://huggingface.co/datasets/next-tat/TAT-QA/resolve/main/tatqa_dataset_dev.json",
            "test": "https://huggingface.co/datasets/next-tat/TAT-QA/resolve/main/tatqa_dataset_test_gold.json",
        }
        raw = load_raw_json_dataset(hf_id, file_urls)
        if raw is None:
            raise Exception("TAT-QA raw JSON loading failed")
        return raw

    # === 特判 2：PubMedQA 配置轮询（方案 A） ===
    if hf_id in ["bigbio/pubmed_qa", "pubmed_qa"]:
        print(f"  🔄 Trying PubMedQA with common configs for {hf_id}")
        # bigbio 版本（任务通常是 bigbio_qa）
        bigbio_configs = [
            "pubmed_qa_pqa_labeled_bigbio_qa",
            "pubmed_qa_pqa_artificial_bigbio_qa",
            "pubmed_qa_pqa_unlabeled_bigbio_qa",
        ]
        # 原版 PubMedQA 的常见配置
        vanilla_configs = [
            "pqa_labeled",
            "pqa_artificial",
            "pqa_unlabeled",
        ]
        cfg_list = bigbio_configs if hf_id == "bigbio/pubmed_qa" else vanilla_configs

        # 先试当前 hf_id 的所有 config
        for cfg in cfg_list:
            try:
                print(f"    • trying config: {cfg}")
                ds = load_dataset(hf_id, cfg)
                print(f"    ✅ loaded: {hf_id} ({cfg})")
                return ds
            except Exception as e:
                print(f"    ✗ failed: {cfg} ({e.__class__.__name__})")

        # 若 bigbio 全失败，回退到原版 pubmed_qa 的 configs
        if hf_id == "bigbio/pubmed_qa":
            for cfg in vanilla_configs:
                try:
                    print(f"    • trying fallback hf_id=pubmed_qa config: {cfg}")
                    ds = load_dataset("pubmed_qa", cfg)
                    print(f"    ✅ loaded: pubmed_qa ({cfg})")
                    return ds
                except Exception:
                    pass

        raise Exception("PubMedQA: all known configs failed")

    # === 其余数据集：原兜底逻辑 ===
    def try_load_single(dataset_id: str):
        # scientific_papers 需要显式 config（根据 hf_id 文本猜测）
        config_name = None
        if dataset_id == "scientific_papers":
            if "arxiv" in (hf_id or "").lower():
                config_name = "arxiv"
            elif "pubmed" in (hf_id or "").lower():
                config_name = "pubmed"

        # 1) 正常加载
        try:
            if config_name:
                return load_dataset(dataset_id, config_name)
            else:
                return load_dataset(dataset_id)
        except (DatasetGenerationError, ValueError, Exception) as e:
            # CUAD 缺少 pdfplumber 的明确提示
            if "theatticusproject/cuad" in dataset_id and "pdfplumber" in str(e).lower():
                raise Exception("CUAD 需要 pdfplumber：请先 `pip install pdfplumber` 然后重试") from e
            pass

        # 2) 流式加载（绕过 Arrow 构建）
        try:
            if config_name:
                return load_dataset(dataset_id, config_name, streaming=True)
            else:
                return load_dataset(dataset_id, streaming=True)
        except (DatasetGenerationError, ValueError, Exception):
            pass

        # 2.5) 流式 + 忽略校验
        try:
            if config_name:
                return load_dataset(dataset_id, config_name, streaming=True, ignore_verifications=True)
            else:
                return load_dataset(dataset_id, streaming=True, ignore_verifications=True)
        except (DatasetGenerationError, ValueError, Exception):
            pass

        # 3) 关闭校验
        try:
            if config_name:
                return load_dataset(dataset_id, config_name, verification_mode="no_checks")
            else:
                return load_dataset(dataset_id, verification_mode="no_checks")
        except Exception:
            pass

        return None

    # 主源
    result = try_load_single(hf_id)
    if result is not None:
        return result

    # 备选源
    if backup_id:
        print(f"  ⚠️  Main source failed, trying backup: {backup_id}")
        result = try_load_single(backup_id)
        if result is not None:
            return result

    # 都失败
    raise Exception(f"Failed to load both {hf_id} and {backup_id or 'no backup'}")



def main():
    parser = ArgumentParser(description="Download datasets to JSONL")
    parser.add_argument("--domain", default=None, help="Filter by domain (finance, government, etc.)")
    parser.add_argument("--dataset", default=None, help="Filter by specific dataset name")
    parser.add_argument("--split", default=None, help="Filter by split (train, validation, test)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip datasets that already exist")
    parser.add_argument("--compress", action="store_true", help="Compress output files with gzip (saves ~70% space but slower to read)")
    parser.add_argument("--list-datasets", action="store_true", help="List all available datasets and exit")
    args = parser.parse_args()

    # 列出所有可用数据集
    if args.list_datasets:
        print("📋 Available datasets:")
        total_datasets = 0
        for domain, datasets in DATASETS.items():
            print(f"\n🏷️  {domain.upper()}:")
            for dataset_name, (hf_id, backup_id) in datasets.items():
                backup_info = f" (备选: {backup_id})" if backup_id else ""
                print(f"   • {dataset_name}: {hf_id}{backup_info}")
                total_datasets += 1
        print(f"\n📊 Total: {total_datasets} datasets across {len(DATASETS)} domains")
        return

    # 下载数据集
    success_count = 0
    error_count = 0
    for domain, datasets in DATASETS.items():
        if args.domain and domain != args.domain:
            continue
        for dataset_name, (hf_id, backup_id) in datasets.items():
            if args.dataset and dataset_name != args.dataset:
                continue
            print(f"Loading {hf_id} → {domain}/{dataset_name}")
            if backup_id:
                print(f"  (备选源: {backup_id})")
            try:
                ds = robust_load_dataset(hf_id, backup_id)
                split_names = list(ds.keys()) if hasattr(ds, "keys") else list(ds)
                for split in split_names:
                    if args.split and split != args.split:
                        continue
                    out_path = os.path.join(
                        BASE_DIR, domain, dataset_name, f"{split}.jsonl"
                    )
                    if args.compress:
                        out_path += ".gz"
                    
                    # 检查是否跳过现有文件
                    if args.skip_existing and os.path.exists(out_path):
                        print(f"  ⏭️  Skipping existing split {split}")
                        continue
                        
                    print(f"  Saving split {split} → {out_path}")
                    try:
                        save_split(ds[split], split, out_path, domain, dataset_name, args.compress)
                        print(f"  ✅ Successfully saved {split} split")
                    except Exception as e:
                        print(f"  ❌ Error saving split {split}: {e}")
                        # 对于Arrow转换错误，尝试直接JSON加载
                        if "Arrow" in str(e) and hf_id in ["next-tat/TAT-QA", "ibm/TAT-QA"]:
                            print(f"  🔄 Retrying with direct JSON loading for {split}")
                            try:
                                file_urls = {
                                    "train": "https://huggingface.co/datasets/next-tat/TAT-QA/resolve/main/tatqa_dataset_train.json",
                                    "validation": "https://huggingface.co/datasets/next-tat/TAT-QA/resolve/main/tatqa_dataset_dev.json",
                                    "test": "https://huggingface.co/datasets/next-tat/TAT-QA/resolve/main/tatqa_dataset_test_gold.json"
                                }
                                raw_ds = load_raw_json_dataset(hf_id, {split: file_urls[split]})
                                if raw_ds and split in raw_ds.keys():
                                    save_split(raw_ds[split], split, out_path, domain, dataset_name, args.compress)
                                    print(f"  ✅ Successfully saved {split} split (via raw JSON)")
                                    continue
                            except Exception as e2:
                                print(f"  ❌ Raw JSON loading also failed: {e2}")
                        error_count += 1
                        continue
                success_count += 1
                print(f"✅ Successfully processed {dataset_name}")
            except Exception as e:
                print(f"❌ Error loading dataset {hf_id}: {e}")
                print(f"   Skipping {dataset_name} and continuing...")
                error_count += 1
                continue
        print(f"\n📊 Summary:")
        print(f"   ✅ Success: {success_count} datasets")
        print(f"   ❌ Errors: {error_count} datasets")
        if error_count > 0:
            print(f"   ⚠️  Some datasets failed to load - this is normal for problematic datasets")

if __name__ == "__main__":
    main()
