# -*- coding: utf-8 -*-
"""
eval/utils.py
人工标注文件处理工具
"""

import json
from pathlib import Path
import pandas as pd


def extract_id_pmid_title_abstract(jsonl_path, save_path=None):
    """
    从人工注释 JSONL 文件中提取 id, pmid, title, abstract.

    Parameters
    ----------
    jsonl_path : str | Path
        输入的 jsonl 文件路径
    save_path : str | Path | None
        可选，如果提供则保存为 CSV 文件

    Returns
    -------
    pd.DataFrame
        包含 [id, pmid, title, abstract] 的 DataFrame
    """
    jsonl_path = Path(jsonl_path)
    records = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            _id = obj.get("id")
            text = obj.get("text", "")
            parts = text.split(":", 2)  # 只切 3 段
            if len(parts) == 3:
                pmid, title, abstract = parts
            else:
                pmid, title, abstract = (None, None, None)

            records.append({
                "id": _id,
                "pmid": pmid,
                "title": title,
                "abstract": abstract
            })

    df = pd.DataFrame(records)

    if save_path:
        save_path = Path(save_path)
        df.to_csv(save_path, index=False, encoding="utf-8-sig")

    return df
