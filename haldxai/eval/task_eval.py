# -*- coding: utf-8 -*-
# haldxai/eval/task_eval.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import pandas as pd
from .eval_utils import (
    load_name2id_map, normalize_and_map, load_gold_by_pmid,
    gold_entities_no_type, count_matches_no_type, prf, build_filename,
    HALD_LABELS, load_name2id_map, load_gold_by_pmid, build_filename, prf,
    normalize_and_map, gold_entities_with_label, count_matches_with_label
)

def load_task_csv(ner_dir: Path, task_name: str) -> pd.DataFrame:
    ner_dir = Path(ner_dir)
    fname = f"annotated_entities_{task_name}.csv" if not task_name.endswith(".csv") else task_name
    p = ner_dir / fname
    if not p.exists():
        raise FileNotFoundError(f"Not found: {p}")
    df = pd.read_csv(p, low_memory=False)
    required = {"pmid", "entity_text", "main_text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{p.name} 缺少列: {missing}")
    df["pmid"] = df["pmid"].astype(str)
    return df

def task_entities_no_type(df_task: pd.DataFrame, name2id: Dict[str,str]) -> Dict[str, List[Dict[str,Any]]]:
    out: Dict[str, List[Dict[str,Any]]] = {}
    for pmid, g in df_task.groupby("pmid", dropna=True):
        pmid = str(pmid)
        items = []
        for _, row in g.iterrows():
            raw = str(row.get("entity_text", "") or "")
            main = str(row.get("main_text", "") or "")
            base = main if main.strip() else raw
            std, ent_id = normalize_and_map(name2id, base)
            if not std:
                continue
            key = ent_id if ent_id is not None else std
            items.append({
                "key": str(key),
                "key_src": "id" if ent_id is not None else "name",
                "text_raw": raw,
                "text_std": std,
                "entity_id": ent_id,
            })
        uniq = {(r["key"], r["text_std"]) for r in items}
        cleaned = []
        for k, s in sorted(uniq):
            for r in items:
                if r["key"] == k and r["text_std"] == s:
                    cleaned.append(r); break
        out[pmid] = cleaned
    return out

def evaluate_task_entities_no_type(
    *, task_name: str, ner_dir: Path, gold_jsonl: Path,
    name2id_json: Path, out_dir: Path, lenient_substr: bool = True
):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df_task = load_task_csv(ner_dir, task_name)
    name2id   = load_name2id_map(name2id_json)
    pmid2gold = load_gold_by_pmid(gold_jsonl)           # gold: pmid -> row
    pmid2pred = task_entities_no_type(df_task, name2id) # task: pmid -> items

    # 仅评估交集 pmid
    pmids_eval = sorted(set(pmid2gold.keys()) & set(pmid2pred.keys()))

    per_rows, rows_fn, rows_fp = [], [], []
    for pmid in pmids_eval:
        gold_items = gold_entities_no_type(pmid2gold[pmid], name2id)
        pred_items = pmid2pred.get(pmid, [])

        tp, fp, fn, gold_only, model_only = count_matches_no_type(pred_items, gold_items, lenient_substr)
        p, r, f = prf(tp, fp, fn)

        per_rows.append({
            "task": task_name, "pmid": pmid,
            "pred_cnt": len(pred_items), "gold_cnt": len(gold_items),
            "tp": tp, "fp": fp, "fn": fn,
            "precision": p, "recall": r, "f1": f,
        })

        gb = {r["key"]: r for r in gold_items}; pb = {r["key"]: r for r in pred_items}
        for k in sorted(gold_only):
            rows_fn.append({"task": task_name, "pmid": pmid, "where": "gold_only(FN)", **gb[k]})
        for k in sorted(model_only):
            rows_fp.append({"task": task_name, "pmid": pmid, "where": "task_only(FP)", **pb[k]})

    policy = "lenient-substr" if lenient_substr else "strict"
    per_df = pd.DataFrame(per_rows)
    per_path = Path(out_dir) / build_filename(task_name, "task-entities_recog", "nolabel", "normid-abbr", policy, "per-article")
    per_df.to_csv(per_path, index=False, encoding="utf-8-sig")

    if len(per_df):
        tp_sum = per_df["tp"].fillna(0).astype(int).sum()
        fp_sum = per_df["fp"].fillna(0).astype(int).sum()
        fn_sum = per_df["fn"].fillna(0).astype(int).sum()
        p_micro, r_micro, f_micro = prf(tp_sum, fp_sum, fn_sum)
        p_macro = per_df["precision"].mean()
        r_macro = per_df["recall"].mean()
        f_macro = per_df["f1"].mean()
    else:
        p_micro = r_micro = f_micro = p_macro = r_macro = f_macro = 0.0
        tp_sum = fp_sum = fn_sum = 0

    sum_df = pd.DataFrame([{
        "task": task_name,
        "articles_evaluated": len(pmids_eval),
        "pred_entities_total": int(per_df["pred_cnt"].fillna(0).sum()) if len(per_df) else 0,
        "gold_entities_total": int(per_df["gold_cnt"].fillna(0).sum()) if len(per_df) else 0,
        "tp_total": int(tp_sum), "fp_total": int(fp_sum), "fn_total": int(fn_sum),
        "precision_micro": p_micro, "recall_micro": r_micro, "f1_micro": f_micro,
        "precision_macro": p_macro, "recall_macro": r_macro, "f1_macro": f_macro,
        "norm_strategy": "normid-abbr", "match_policy": policy, "label_mode": "nolabel",
        "pmids_eval_count": len(pmids_eval),
    }])
    sum_path = Path(out_dir) / build_filename(task_name, "task-entities_recog", "nolabel","normid-abbr", policy, "summary")
    sum_df.to_csv(sum_path, index=False, encoding="utf-8-sig")

    fn_df = pd.DataFrame(rows_fn); fp_df = pd.DataFrame(rows_fp)
    fn_path = Path(out_dir) / build_filename(task_name, "task-entities_recog","nolabel","normid-abbr", policy, "FN")
    fp_path = Path(out_dir) / build_filename(task_name, "task-entities_recog","nolabel","normid-abbr", policy, "FP")
    fn_df.to_csv(fn_path, index=False, encoding="utf-8-sig")
    fp_df.to_csv(fp_path, index=False, encoding="utf-8-sig")

    return {"per_article": per_path, "summary": sum_path, "fn": fn_path, "fp": fp_path}

def batch_evaluate_tasks_no_type(
    *, task_names: List[str], ner_dir: Path, gold_jsonl: Path,
    name2id_json: Path, out_dir: Path, lenient_substr: bool = True
) -> pd.DataFrame:
    rows = []
    for tn in task_names:
        paths = evaluate_task_entities_no_type(
            task_name=tn, ner_dir=ner_dir, gold_jsonl=gold_jsonl,
            name2id_json=name2id_json, out_dir=out_dir, lenient_substr=lenient_substr
        )
        rows.append(pd.read_csv(paths["summary"]).iloc[0].to_dict())
    policy = "lenient-substr" if lenient_substr else "strict"
    all_df = pd.DataFrame(rows)
    all_df.to_csv(Path(out_dir)/f"_ALL_task-entities_recog-nolabel_normid-abbr_{policy}_summary.csv",
                  index=False, encoding="utf-8-sig")
    return all_df

def task_entities_with_label(
    df_task: pd.DataFrame, name2id: Dict[str,str],
    allowed_labels: Optional[List[str]] = HALD_LABELS
) -> Dict[str, List[Dict[str,Any]]]:
    out: Dict[str, List[Dict[str,Any]]] = {}
    for pmid, g in df_task.groupby("pmid", dropna=True):
        pmid = str(pmid)
        items = []
        for _, row in g.iterrows():
            raw  = str(row.get("entity_text", "") or "")
            main = str(row.get("main_text", "") or "")
            base = main if main.strip() else raw
            std, ent_id = normalize_and_map(name2id, base)
            if not std:
                continue
            etype = str(row.get("entity_type","")).strip().upper()
            if allowed_labels is not None and etype not in allowed_labels:
                continue
            key = ent_id if ent_id is not None else std
            items.append({"key": str(key), "type": etype,
                          "key_src": "id" if ent_id is not None else "name",
                          "text_raw": raw, "text_std": std, "entity_id": ent_id})
        uniq = {(r["key"], r["type"], r["text_std"]) for r in items}
        cleaned = []
        for k,t,s in sorted(uniq):
            for r in items:
                if r["key"] == k and r["type"] == t and r["text_std"] == s:
                    cleaned.append(r); break
        out[pmid] = cleaned
    return out

def evaluate_task_entities_with_label(
    *, task_name: str, ner_dir: Path, gold_jsonl: Path,
    name2id_json: Path, out_dir: Path,
    lenient_substr: bool = True, allowed_labels: Optional[List[str]] = HALD_LABELS
):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    # 读 task csv（pmid 交集的逻辑仍保持）
    def _load_task_csv(ner_dir: Path, task_name: str) -> pd.DataFrame:
        ner_dir = Path(ner_dir)
        fname = f"annotated_entities_{task_name}.csv" if not task_name.endswith(".csv") else task_name
        p = ner_dir / fname
        if not p.exists(): raise FileNotFoundError(f"Not found: {p}")
        df = pd.read_csv(p, low_memory=False)
        for col in ["pmid","entity_text","main_text","entity_type"]:
            if col not in df.columns:
                raise ValueError(f"{p.name} 缺少列: {col}")
        df["pmid"] = df["pmid"].astype(str)
        return df

    df_task = _load_task_csv(ner_dir, task_name)
    name2id   = load_name2id_map(name2id_json)
    pmid2gold = load_gold_by_pmid(gold_jsonl)                             # gold: pmid -> row
    pmid2pred = task_entities_with_label(df_task, name2id, allowed_labels) # task: pmid -> items

    pmids_eval = sorted(set(pmid2gold.keys()) & set(pmid2pred.keys()))

    per_rows, rows_fn, rows_fp = [], [], []
    label_counters = {lab: {"tp":0, "fp":0, "fn":0, "pred":0, "gold":0} for lab in allowed_labels}

    for pmid in pmids_eval:
        gold_items = gold_entities_with_label(pmid2gold[pmid], name2id, allowed_labels)
        pred_items = pmid2pred.get(pmid, [])

        tp, fp, fn, gold_only, model_only = count_matches_with_label(pred_items, gold_items, lenient_substr)
        p, r, f = prf(tp, fp, fn)

        # per-label 细分
        detail = []
        for lab in allowed_labels:
            pe = [x for x in pred_items if x["type"] == lab]
            ge = [x for x in gold_items if x["type"] == lab]
            ttp, tfp, tfn, _, _ = count_matches_with_label(pe, ge, lenient_substr)
            lp, lr, lf = prf(ttp, tfp, tfn)
            detail.append({"label": lab, "tp": ttp, "fp": tfp, "fn": tfn,
                           "precision": lp, "recall": lr, "f1": lf,
                           "pred_cnt": len(pe), "gold_cnt": len(ge)})
            label_counters[lab]["tp"]   += ttp
            label_counters[lab]["fp"]   += tfp
            label_counters[lab]["fn"]   += tfn
            label_counters[lab]["pred"] += len(pe)
            label_counters[lab]["gold"] += len(ge)

        per_rows.append({"task": task_name, "pmid": pmid,
                         "pred_cnt": len(pred_items), "gold_cnt": len(gold_items),
                         "tp": tp, "fp": fp, "fn": fn,
                         "precision": p, "recall": r, "f1": f,
                         "detail_by_label": json.dumps(detail, ensure_ascii=False)})

        gb = {(r["key"], r["type"]): r for r in gold_items}
        pb = {(r["key"], r["type"]): r for r in pred_items}
        for pair in sorted(gold_only):
            rows_fn.append({"task": task_name, "pmid": pmid, "where": "gold_only(FN)", **gb[pair]})
        for pair in sorted(model_only):
            rows_fp.append({"task": task_name, "pmid": pmid, "where": "task_only(FP)", **pb[pair]})

    policy = "lenient-substr" if lenient_substr else "strict"
    per_df = pd.DataFrame(per_rows)
    per_path = Path(out_dir) / build_filename(task_name, "task-entities_recog", "withlabel", "normid-abbr", policy, "per-article")
    per_df.to_csv(per_path, index=False, encoding="utf-8-sig")

    # summary（整体）
    if len(per_df):
        tp_sum = per_df["tp"].fillna(0).astype(int).sum()
        fp_sum = per_df["fp"].fillna(0).astype(int).sum()
        fn_sum = per_df["fn"].fillna(0).astype(int).sum()
        p_micro, r_micro, f_micro = prf(tp_sum, fp_sum, fn_sum)
        p_macro = per_df["precision"].mean()
        r_macro = per_df["recall"].mean()
        f_macro = per_df["f1"].mean()
    else:
        p_micro = r_micro = f_micro = p_macro = r_macro = f_macro = 0.0
        tp_sum = fp_sum = fn_sum = 0

    sum_df = pd.DataFrame([{
        "task": task_name,
        "articles_evaluated": len(pmids_eval),
        "tp_total": int(tp_sum), "fp_total": int(fp_sum), "fn_total": int(fn_sum),
        "precision_micro": p_micro, "recall_micro": r_micro, "f1_micro": f_micro,
        "precision_macro": p_macro, "recall_macro": r_macro, "f1_macro": f_macro,
        "norm_strategy": "normid-abbr", "match_policy": policy, "label_mode": "withlabel",
        "pmids_eval_count": len(pmids_eval),
    }])
    sum_path = Path(out_dir) / build_filename(task_name, "task-entities_recog", "withlabel", "normid-abbr", policy, "summary")
    sum_df.to_csv(sum_path, index=False, encoding="utf-8-sig")

    # summary_by_label
    rows_lab = []
    for lab, c in label_counters.items():
        lp, lr, lf = prf(c["tp"], c["fp"], c["fn"])
        rows_lab.append({"label": lab, "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                         "pred_total": c["pred"], "gold_total": c["gold"],
                         "precision": lp, "recall": lr, "f1": lf})
    lbl_df = pd.DataFrame(rows_lab)
    lbl_path = Path(out_dir) / build_filename(task_name, "task-entities_recog", "withlabel", "normid-abbr", policy, "summary_by_label")
    lbl_df.to_csv(lbl_path, index=False, encoding="utf-8-sig")

    # 明细
    fn_df = pd.DataFrame(rows_fn); fp_df = pd.DataFrame(rows_fp)
    fn_path = Path(out_dir) / build_filename(task_name, "task-entities_recog","withlabel","normid-abbr", policy, "FN")
    fp_path = Path(out_dir) / build_filename(task_name, "task-entities_recog","withlabel","normid-abbr", policy, "FP")
    fn_df.to_csv(fn_path, index=False, encoding="utf-8-sig")
    fp_df.to_csv(fp_path, index=False, encoding="utf-8-sig")

    return {"per_article": per_path, "summary": sum_path, "summary_by_label": lbl_path, "fn": fn_path, "fp": fp_path}

def batch_evaluate_tasks_with_label(
    *, task_names: List[str], ner_dir: Path, gold_jsonl: Path,
    name2id_json: Path, out_dir: Path, lenient_substr: bool = True,
    allowed_labels: Optional[List[str]] = HALD_LABELS
) -> pd.DataFrame:
    rows = []
    for tn in task_names:
        paths = evaluate_task_entities_with_label(
            task_name=tn, ner_dir=ner_dir, gold_jsonl=gold_jsonl,
            name2id_json=name2id_json, out_dir=out_dir,
            lenient_substr=lenient_substr, allowed_labels=allowed_labels
        )
        rows.append(pd.read_csv(paths["summary"]).iloc[0].to_dict())
    policy = "lenient-substr" if lenient_substr else "strict"
    all_df = pd.DataFrame(rows)
    all_df.to_csv(Path(out_dir)/f"_ALL_task-entities_recog-withlabel_normid-abbr_{policy}_summary.csv",
                  index=False, encoding="utf-8-sig")
    return all_df
