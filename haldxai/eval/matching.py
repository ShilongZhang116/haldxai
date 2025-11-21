# -*- coding: utf-8 -*-
"""
haldxai/eval/matching.py

用途：
1) 读取人工标注摘要元信息 CSV（含 id/pmid/title/abstract）
2) 批量读取 base-llm_anno_raw 目录下各模型文件（.json/.jsonl）
   —— 文件内部可能是 JSON 列表，或 JSONL（每行一个 JSON 对象）
3) 对于每个模型样本，尝试匹配其对应的 (id, pmid)
   匹配策略：
     a) 若样本自带 'abstract' 或 'text' 字段，先做规范化后精确匹配；
     b) 否则使用 'entities' 与 'relationships' 中的 evidence 片段构造候选，
        在 meta['abstract'] 中做包含匹配，采用“命中次数 + 片段长度”评分。
4) 输出合并后的匹配结果 CSV：每行包含 [model, sample_idx, id, pmid, score, matched_title]

注意：
- 兼容 JSON 列表与 JSONL 两种存储格式；
- 文本规范化会去除多余空白与标点差异，以提高匹配鲁棒性。
"""

from __future__ import annotations
import json, re, hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd


# --------------------------- 文本规范化与指纹 ---------------------------

_ws_re = re.compile(r"\s+", re.UNICODE)
_punct_re = re.compile(r"[^\w]+", re.UNICODE)

def _normalize_text(s: str) -> str:
    """粗规范化：小写、压缩空白、去掉左右空白。"""
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = _ws_re.sub(" ", s)
    return s

def _fingerprint(s: str) -> str:
    """更强规范化后做指纹：移除非字母数字（含下划线），再 sha1。"""
    s = _normalize_text(s)
    s = _punct_re.sub("", s)
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


# --------------------------- 文件读取助手 ---------------------------

def _load_json_maybe_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    读取模型文件：
    - 若是一个 JSON 列表，直接返回；
    - 若是 JSONL（每行一个 JSON），读取为列表；
    """
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []
    if text.startswith("["):
        data = json.loads(text)
        if isinstance(data, list):
            return data
        raise ValueError(f"{path} 顶层不是列表")
    # JSONL
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # 处理可能的 BOM
        if line and line[0] == "\ufeff":
            line = line.lstrip("\ufeff")
        try:
            obj = json.loads(line)
            out.append(obj)
        except json.JSONDecodeError:
            # 跳过坏行
            continue
    return out


# --------------------------- 元信息加载 ---------------------------

def load_admin_meta(meta_csv: Path) -> pd.DataFrame:
    """
    读取人工标注摘要元信息 CSV，要求包含列：id, pmid, title, abstract
    预先计算规范化字段与指纹以便快速匹配。
    """
    df = pd.read_csv(meta_csv, dtype=str, low_memory=False)
    # 强制列存在
    for col in ["id", "pmid", "title", "abstract"]:
        if col not in df.columns:
            raise ValueError(f"缺少列：{col}")
    # 规范化文本与指纹
    df["_abstract_norm"] = df["abstract"].fillna("").map(_normalize_text)
    df["_abstract_fp"] = df["_abstract_norm"].map(_fingerprint)
    return df


# --------------------------- 匹配核心 ---------------------------

def _extract_candidate_snippets(sample: Dict[str, Any], min_len: int = 40, top_k: int = 8) -> List[str]:
    """
    从样本中提取可用于匹配的证据片段（entities/relationships 的 evidence 字段）。
    仅取长度 >= min_len 的片段，按长度降序保留 top_k 个。
    """
    snippets = []
    for key in ("entities", "relationships"):
        items = sample.get(key, [])
        if isinstance(items, list):
            for it in items:
                ev = it.get("evidence") if isinstance(it, dict) else None
                if isinstance(ev, str):
                    s = _normalize_text(ev)
                    if len(s) >= min_len:
                        snippets.append(s)
    # 去重、按长度排序
    uniq = sorted(set(snippets), key=lambda x: (-len(x), x))
    return uniq[:top_k]

def _exact_match_by_text(sample: Dict[str, Any], meta_df: pd.DataFrame) -> Optional[Tuple[str, str, str, float]]:
    """
    若样本中存在 'abstract' 或 'text' 字段，则进行精确匹配（指纹一致）。
    返回 (id, pmid, title, score) 或 None
    """
    for k in ("abstract", "text"):
        if k in sample and isinstance(sample[k], str):
            norm = _normalize_text(sample[k])
            if not norm:
                continue
            fp = _fingerprint(norm)
            m = meta_df.loc[meta_df["_abstract_fp"] == fp]
            if len(m) == 1:
                r = m.iloc[0]
                return (str(r["id"]), str(r["pmid"]), str(r["title"]), 1.0)
            # 若有多条同指纹（极少见），返回 None 让模糊匹配接管
    return None

def _fuzzy_match_by_snippets(sample: Dict[str, Any], meta_df: pd.DataFrame) -> Optional[Tuple[str, str, str, float]]:
    """
    使用 evidence 片段在 meta_df['_abstract_norm'] 中做包含匹配。
    得分 = 命中片段数 * 1000 + 最长命中片段长度。
    返回 (id, pmid, title, score) 或 None
    """
    snippets = _extract_candidate_snippets(sample)
    if not snippets:
        return None

    scores = []
    abstracts = meta_df["_abstract_norm"].tolist()

    for idx, abs_norm in enumerate(abstracts):
        hits = 0
        best_len = 0
        for snip in snippets:
            if snip in abs_norm:
                hits += 1
                if len(snip) > best_len:
                    best_len = len(snip)
        if hits > 0:
            score = hits * 1000 + best_len
            scores.append((idx, score, hits, best_len))

    if not scores:
        return None

    # 选择分数最高，若并列，优先 hits 多者，再看 best_len
    scores.sort(key=lambda x: (-x[1], -x[2], -x[3], x[0]))
    best_idx, best_score, _, _ = scores[0]
    row = meta_df.iloc[best_idx]
    return (str(row["id"]), str(row["pmid"]), str(row["title"]), float(best_score))


def match_one_sample(sample: Dict[str, Any], meta_df: pd.DataFrame) -> Optional[Tuple[str, str, str, float]]:
    """
    匹配单个样本，优先精确匹配，其次模糊（evidence）匹配。
    返回 (id, pmid, title, score) 或 None
    """
    exact = _exact_match_by_text(sample, meta_df)
    if exact is not None:
        return exact
    return _fuzzy_match_by_snippets(sample, meta_df)


# --------------------------- 批处理入口 ---------------------------

def match_model_file(model_path: Path, meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    对单个模型文件做匹配，返回 DataFrame：
    [model, sample_idx, id, pmid, title, score]
    """
    model_path = Path(model_path)
    model_name = model_path.stem  # 文件名作为模型名
    data = _load_json_maybe_jsonl(model_path)

    rows = []
    for i, sample in enumerate(data):
        res = match_one_sample(sample, meta_df)
        if res is None:
            rows.append({"model": model_name, "sample_idx": i, "id": None, "pmid": None, "title": None, "score": 0.0})
        else:
            _id, _pmid, _title, _score = res
            rows.append({"model": model_name, "sample_idx": i, "id": _id, "pmid": _pmid, "title": _title, "score": _score})

    return pd.DataFrame(rows)


def batch_match_models(models_dir: Path, meta_csv: Path, out_csv: Optional[Path] = None) -> pd.DataFrame:
    """
    批量匹配目录下所有 .json / .jsonl 模型文件，汇总输出。
    """
    models_dir = Path(models_dir)
    meta_df = load_admin_meta(Path(meta_csv))

    files = sorted(list(models_dir.glob("*.json"))) + sorted(list(models_dir.glob("*.jsonl")))
    all_rows = []
    for f in files:
        df_one = match_model_file(f, meta_df)
        all_rows.append(df_one)
    result = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(columns=["model","sample_idx","id","pmid","title","score"])

    if out_csv is not None:
        out_csv = Path(out_csv)
        result.to_csv(out_csv, index=False, encoding="utf-8-sig")
    return result

