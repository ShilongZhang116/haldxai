# -*- coding: utf-8 -*-
# haldxai/eval/eval_utils.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Set
import json, re
import pandas as pd
from haldxai.ner.abbrev import parse_entity_abbreviation

# ---------------- 基础工具 ----------------
_ws_re     = re.compile(r"\s+")
_edge_punc = re.compile(r"^[\W_]+|[\W_]+$")
_digit     = re.compile(r"(\d+)")

HALD_LABELS: List[str] = ["BMC", "EGR", "ASPKM", "CRD", "APP", "SCN", "AAI", "CRBC", "NM", "EF"]

def norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = _ws_re.sub(" ", s)
    s = _edge_punc.sub("", s)
    return s

def only_digits(s: str) -> Optional[str]:
    if s is None:
        return None
    m = _digit.search(str(s))
    return m.group(1) if m else None

def load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    txt = Path(path).read_text(encoding="utf-8", errors="replace").strip()
    if not txt:
        return []
    if txt.lstrip().startswith("["):
        data = json.loads(txt)
        if isinstance(data, list):
            return data
        raise ValueError(f"{path} 顶层不是列表")
    out = []
    for line in txt.splitlines():
        line = line.strip()
        if not line:
            continue
        if line and line[0] == "\ufeff":
            line = line.lstrip("\ufeff")
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out

def safe_slice(txt: str, s, t) -> str:
    try:
        s = int(s); t = int(t)
    except Exception:
        return ""
    if s is None or t is None: return ""
    if s < 0: s = 0
    if t < 0: t = 0
    if s > t: s, t = t, s
    if s >= len(txt): return ""
    if t > len(txt): t = len(txt)
    return txt[s:t]

def parse_text_to_meta(text_field: str) -> Tuple[Optional[str], str, str]:
    """Gold text: <pmid>:<title>:<abstract>"""
    if not isinstance(text_field, str):
        return None, "", ""
    parts = text_field.split(":", 2)
    if len(parts) < 3:
        return None, "", text_field
    return only_digits(parts[0]), parts[1], parts[2]

def build_filename(
    model: str,
    task: str = "entities_recog",
    label_mode: str = "nolabel",       # or "withlabel"
    norm_strategy: str = "normid-abbr",
    match_policy: str = "lenient-substr",
    suffix: str = "per-article",
    ext: str = "csv",
) -> str:
    return f"{model}_{task}-{label_mode}_{norm_strategy}_{match_policy}_{suffix}.{ext}"

# ---------------- 归一化 & name->id ----------------
def load_name2id_map(path: Path) -> Dict[str, str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    m: Dict[str, str] = {}
    for k, v in raw.items():
        nk = norm_name(str(k))
        if nk and v not in (None, ""):
            m[nk] = str(v)
    return m

def normalize_and_map(name2id: Dict[str, str], text: str) -> Tuple[str, Optional[str]]:
    try:
        main = parse_entity_abbreviation(text).get("main_text", text)
    except Exception:
        main = text
    std = norm_name(main)
    ent_id = name2id.get(std) if std else None
    return std, ent_id

# ---------------- 指标 ----------------
def prf(tp: int, fp: int, fn: int) -> Tuple[float,float,float]:
    p = tp / (tp + fp) if (tp+fp)>0 else 0.0
    r = tp / (tp + fn) if (tp+fn)>0 else 0.0
    f = 2*p*r/(p+r) if (p+r)>0 else 0.0
    return p, r, f

# ---------------- 匹配（忽略类型） ----------------
def count_matches_no_type(
    pred_items: List[Dict[str,Any]],
    gold_items: List[Dict[str,Any]],
    lenient_substr: bool = True
) -> Tuple[int,int,int,Set[str],Set[str]]:
    pred_keys = {r["key"] for r in pred_items}
    gold_keys = {r["key"] for r in gold_items}
    matched = pred_keys & gold_keys
    gold_only = gold_keys - matched
    model_only = pred_keys - matched

    if lenient_substr:
        gold_name = {r["key"]: r for r in gold_items if r["key_src"] == "name" and r["key"] in gold_only}
        pred_name = {r["key"]: r for r in pred_items if r["key_src"] == "name" and r["key"] in model_only}
        rm_g, rm_p = set(), set()
        for gk, gv in gold_name.items():
            for pk, pv in pred_name.items():
                if gv["text_std"] in pv["text_std"] or pv["text_std"] in gv["text_std"]:
                    rm_g.add(gk); rm_p.add(pk); break
        gold_only -= rm_g
        model_only -= rm_p
        matched |= (rm_g | rm_p)

    tp = len(matched); fp = len(model_only); fn = len(gold_only)
    return tp, fp, fn, gold_only, model_only

# ---------------- Gold/Pred 抽取（忽略类型） ----------------
def gold_entities_no_type(gold_row: Dict[str, Any], name2id: Dict[str, str]) -> List[Dict[str, Any]]:
    full_text = gold_row.get("text", "") or ""
    res = []
    for e in gold_row.get("entities", []):
        raw = safe_slice(full_text, e.get("start_offset"), e.get("end_offset"))
        std, ent_id = normalize_and_map(name2id, raw)
        if not std:
            continue
        key = ent_id if ent_id is not None else std
        res.append({"key": str(key), "key_src": "id" if ent_id is not None else "name",
                    "text_raw": raw, "text_std": std, "entity_id": ent_id})
    uniq = {(r["key"], r["text_std"]) for r in res}
    out = []
    for k, s in sorted(uniq):
        for r in res:
            if r["key"] == k and r["text_std"] == s:
                out.append(r); break
    return out

def model_entities_no_type(sample: Dict[str, Any], name2id: Dict[str, str]) -> List[Dict[str, Any]]:
    res = []
    for e in sample.get("entities", []):
        raw = e.get("entity_text", "")
        std, ent_id = normalize_and_map(name2id, raw)
        if not std:
            continue
        key = ent_id if ent_id is not None else std
        res.append({"key": str(key), "key_src": "id" if ent_id is not None else "name",
                    "text_raw": raw, "text_std": std, "entity_id": ent_id})
    uniq = {(r["key"], r["text_std"]) for r in res}
    out = []
    for k, s in sorted(uniq):
        for r in res:
            if r["key"] == k and r["text_std"] == s:
                out.append(r); break
    return out

# =========================================================
# ==============  以下：带“类型”的版本  ===================
# =========================================================

# ---- 匹配（考虑类型） ----
def count_matches_with_label(
    pred_items: List[Dict[str,Any]],
    gold_items: List[Dict[str,Any]],
    lenient_substr: bool = True
) -> Tuple[int,int,int,Set[Tuple[str,str]],Set[Tuple[str,str]]]:
    """
    item: {"key","type","key_src","text_std",...}
    1) 严格：配对键 (key, type) 完全一致
    2) 宽松补充：仅对 key_src=='name' 的剩余项，若 type 相同且 name 包含关系成立则视为匹配
    返回：TP, FP, FN, gold_only_pairs, model_only_pairs
    """
    pred_pairs = {(r["key"], r["type"]) for r in pred_items}
    gold_pairs = {(r["key"], r["type"]) for r in gold_items}
    matched = pred_pairs & gold_pairs
    gold_only = gold_pairs - matched
    model_only = pred_pairs - matched

    if lenient_substr:
        go = {(r["key"], r["type"]): r for r in gold_items
              if (r["key"], r["type"]) in gold_only and r["key_src"] == "name"}
        mo = {(r["key"], r["type"]): r for r in pred_items
              if (r["key"], r["type"]) in model_only and r["key_src"] == "name"}
        rm_g, rm_p = set(), set()
        for (gk, gt), gv in go.items():
            for (pk, pt), pv in mo.items():
                if gt != pt:    # 类型必须一致
                    continue
                if gv["text_std"] in pv["text_std"] or pv["text_std"] in gv["text_std"]:
                    rm_g.add((gk, gt)); rm_p.add((pk, pt)); break
        gold_only -= rm_g
        model_only -= rm_p
        matched |= (rm_g | rm_p)

    tp = len(matched); fp = len(model_only); fn = len(gold_only)
    return tp, fp, fn, gold_only, model_only

# ---- Gold/Pred 抽取（带类型） ----
def _canon_label(x: Any, allowed: Optional[List[str]]) -> Optional[str]:
    if not isinstance(x, str):
        return None
    lab = x.strip().upper()
    if allowed is None or lab in allowed:
        return lab
    return None  # 不在白名单里就丢弃

def gold_entities_with_label(
    gold_row: Dict[str, Any], name2id: Dict[str, str],
    allowed_labels: Optional[List[str]] = HALD_LABELS
) -> List[Dict[str, Any]]:
    full_text = gold_row.get("text", "") or ""
    res = []
    for e in gold_row.get("entities", []):
        etype = _canon_label(e.get("label", ""), allowed_labels)
        if not etype:
            continue
        raw = safe_slice(full_text, e.get("start_offset"), e.get("end_offset"))
        std, ent_id = normalize_and_map(name2id, raw)
        if not std:
            continue
        key = ent_id if ent_id is not None else std
        res.append({"key": str(key), "type": etype,
                    "key_src": "id" if ent_id is not None else "name",
                    "text_raw": raw, "text_std": std, "entity_id": ent_id})
    # 去重
    uniq = {(r["key"], r["type"], r["text_std"]) for r in res}
    out = []
    for k, t, s in sorted(uniq):
        for r in res:
            if r["key"] == k and r["type"] == t and r["text_std"] == s:
                out.append(r); break
    return out

def model_entities_with_label(
    sample: Dict[str, Any], name2id: Dict[str, str],
    allowed_labels: Optional[List[str]] = HALD_LABELS
) -> List[Dict[str, Any]]:
    res = []
    for e in sample.get("entities", []):
        etype = _canon_label(e.get("entity_type", ""), allowed_labels)
        if not etype:
            continue
        raw = e.get("entity_text", "")
        std, ent_id = normalize_and_map(name2id, raw)
        if not std:
            continue
        key = ent_id if ent_id is not None else std
        res.append({"key": str(key), "type": etype,
                    "key_src": "id" if ent_id is not None else "name",
                    "text_raw": raw, "text_std": std, "entity_id": ent_id})
    uniq = {(r["key"], r["type"], r["text_std"]) for r in res}
    out = []
    for k, t, s in sorted(uniq):
        for r in res:
            if r["key"] == k and r["type"] == t and r["text_std"] == s:
                out.append(r); break
    return out

# ---- Gold（按 pmid） ----
def load_gold_by_pmid(gold_jsonl: Path) -> Dict[str, Dict[str, Any]]:
    pmid2row: Dict[str, Dict[str, Any]] = {}
    with open(gold_jsonl, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            pmid, _, _ = parse_text_to_meta(obj.get("text", ""))
            if pmid:
                pmid2row[str(pmid)] = obj
    return pmid2row
