from pathlib import Path
import json

root = Path("data/chunked")  # 你的 chunked 文件夹路径
count = 0

for p in root.rglob("*.jsonl"):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if "text" in obj and obj["text"].strip():
                count += 1

print(f"总 chunk 数量: {count}")
