import json, os
from pathlib import Path
import pandas as pd
from uuid import uuid4

def csv_to_jsonl(csv_path: Path, jsonl_path: Path, system_prompt: str, model_name: str):
    df = pd.read_csv(csv_path)
    pmids, entries = [], []

    for row in df.itertuples():
        pmid   = str(getattr(row, "pmid"))  # 或 row.pmid
        qid    = f"{pmid}_{uuid4().hex[:8]}"   # ▶️ 唯一 ID：PMID_随机 8 位
        prompt = getattr(row, "abstract")

        req = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": prompt}
            ],
            "request_id": qid,     # 🚀 自定义字段
            "user": pmid           # （可选）OpenAI 官方字段，用来追踪 End-User
        }
        entries.append(req)
        pmids.append(pmid)

    # —— 写 .jsonl —— #
    with jsonl_path.open("w", encoding="utf-8") as fout:
        for req in entries:
            fout.write(json.dumps(req, ensure_ascii=False) + "\n")

    # —— 额外生成一张「索引表」—— #
    idx_csv = jsonl_path.with_suffix(".meta.csv")
    pd.DataFrame({"pmid": pmids, "request_id": [e["request_id"] for e in entries]}
                 ).to_csv(idx_csv, index=False, encoding="utf-8-sig")
    print(f"✅ {jsonl_path.name} 以及索引表已生成")
