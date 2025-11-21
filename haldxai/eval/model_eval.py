# -*- coding: utf-8 -*-
# haldxai/eval/model_eval.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import pandas as pd
from .eval_utils import (
    load_json_or_jsonl, load_name2id_map, model_entities_no_type,
    gold_entities_no_type, count_matches_no_type, prf, build_filename,
    HALD_LABELS, load_json_or_jsonl, load_name2id_map, build_filename, prf,
    gold_entities_with_label, model_entities_with_label, count_matches_with_label
)


def list_model_names(models_dir: Path) -> List[str]:
    names = set()
    for p in Path(models_dir).glob("*.json"):  names.add(p.stem)
    for p in Path(models_dir).glob("*.jsonl"): names.add(p.stem)
    return sorted(names)

def evaluate_entities_no_type(
    *, model_name: str, models_dir: Path, match_csv: Path,
    gold_jsonl: Path, name2id_json: Path, out_dir: Path,
    lenient_substr: bool = True
) -> Dict[str, Path]:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # sample_idx -> gold id/pmid/title
    match_df = pd.read_csv(match_csv, dtype=str)
    m_map = (match_df.query("model == @model_name")
             .dropna(subset=["id"])
             .assign(sample_idx=lambda d: d["sample_idx"].astype(int))
             .set_index("sample_idx"))

    # gold: id -> row
    gold_rows = {}
    with open(gold_jsonl, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                gold_rows[str(obj.get("id"))] = obj
            except json.JSONDecodeError:
                pass

    # 模型文件
    mf = Path(models_dir) / f"{model_name}.json"
    if not mf.exists():
        mf = Path(models_dir) / f"{model_name}.jsonl"
    samples = load_json_or_jsonl(mf)

    name2id = load_name2id_map(name2id_json)

    per_rows, rows_fn, rows_fp = [], [], []
    for idx, sample in enumerate(samples):
        if idx not in m_map.index:
            pred_items = model_entities_no_type(sample, name2id)
            per_rows.append({"model": model_name, "sample_idx": idx, "id": None,
                             "pmid": None, "title": None, "pred_cnt": len(pred_items),
                             "gold_cnt": None, "tp": 0, "fp": None, "fn": None,
                             "precision": 0.0, "recall": 0.0, "f1": 0.0})
            continue

        rid = str(m_map.loc[idx, "id"]); pmid = m_map.loc[idx].get("pmid"); title = m_map.loc[idx].get("title")
        gold_obj = gold_rows.get(rid)
        pred_items = model_entities_no_type(sample, name2id)

        if gold_obj is None:
            tp, fp, fn = 0, len(pred_items), 0
            p, r, f = prf(tp, fp, fn)
            per_rows.append({"model": model_name, "sample_idx": idx, "id": rid, "pmid": pmid, "title": title,
                             "pred_cnt": len(pred_items), "gold_cnt": 0,
                             "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f})
            for it in pred_items:
                rows_fp.append({"model": model_name, "sample_idx": idx, "id": rid, "pmid": pmid, "title": title,
                                "where": "model_only(FP)", **it})
            continue

        gold_items = gold_entities_no_type(gold_obj, name2id)
        tp, fp, fn, gold_only, model_only = count_matches_no_type(pred_items, gold_items, lenient_substr)
        p, r, f = prf(tp, fp, fn)

        per_rows.append({"model": model_name, "sample_idx": idx, "id": rid, "pmid": pmid, "title": title,
                         "pred_cnt": len(pred_items), "gold_cnt": len(gold_items),
                         "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f})

        gb = {r["key"]: r for r in gold_items}; pb = {r["key"]: r for r in pred_items}
        for k in sorted(gold_only):
            rows_fn.append({"model": model_name, "sample_idx": idx, "id": rid, "pmid": pmid, "title": title,
                            "where": "gold_only(FN)", **gb[k]})
        for k in sorted(model_only):
            rows_fp.append({"model": model_name, "sample_idx": idx, "id": rid, "pmid": pmid, "title": title,
                            "where": "model_only(FP)", **pb[k]})

    policy = "lenient-substr" if lenient_substr else "strict"
    per_df = pd.DataFrame(per_rows)
    per_path = Path(out_dir) / build_filename(model_name, "entities_recog", "nolabel", "normid-abbr", policy, "per-article")
    per_df.to_csv(per_path, index=False, encoding="utf-8-sig")

    tp_sum = per_df["tp"].fillna(0).astype(int).sum()
    fp_sum = per_df["fp"].fillna(0).astype(int).sum()
    fn_sum = per_df["fn"].fillna(0).astype(int).sum()
    p_micro, r_micro, f_micro = prf(tp_sum, fp_sum, fn_sum)
    p_macro = per_df["precision"].mean(); r_macro = per_df["recall"].mean(); f_macro = per_df["f1"].mean()

    sum_df = pd.DataFrame([{
        "model": model_name, "articles_evaluated": int(per_df["id"].notna().sum()),
        "pred_entities_total": int(per_df["pred_cnt"].fillna(0).sum()),
        "gold_entities_total": int(per_df["gold_cnt"].fillna(0).sum()),
        "tp_total": int(tp_sum), "fp_total": int(fp_sum), "fn_total": int(fn_sum),
        "precision_micro": p_micro, "recall_micro": r_micro, "f1_micro": f_micro,
        "precision_macro": p_macro, "recall_macro": r_macro, "f1_macro": f_macro,
        "norm_strategy": "normid-abbr", "match_policy": policy, "label_mode": "nolabel",
    }])
    sum_path = Path(out_dir) / build_filename(model_name, "entities_recog", "nolabel","normid-abbr", policy, "summary")
    sum_df.to_csv(sum_path, index=False, encoding="utf-8-sig")

    fn_df = pd.DataFrame(rows_fn); fp_df = pd.DataFrame(rows_fp)
    fn_path = Path(out_dir) / build_filename(model_name, "entities_recog","nolabel","normid-abbr", policy, "FN")
    fp_path = Path(out_dir) / build_filename(model_name, "entities_recog","nolabel","normid-abbr", policy, "FP")
    fn_df.to_csv(fn_path, index=False, encoding="utf-8-sig")
    fp_df.to_csv(fp_path, index=False, encoding="utf-8-sig")

    return {"per_article": per_path, "summary": sum_path, "fn": fn_path, "fp": fp_path}

def batch_evaluate_entities_no_type(
    *, models_dir: Path, match_csv: Path, gold_jsonl: Path,
    name2id_json: Path, out_dir: Path, model_names: Optional[List[str]]=None,
    lenient_substr: bool = True
) -> pd.DataFrame:
    if model_names is None:
        model_names = list_model_names(models_dir)
    rows = []
    for mn in model_names:
        paths = evaluate_entities_no_type(
            model_name=mn, models_dir=models_dir, match_csv=match_csv,
            gold_jsonl=gold_jsonl, name2id_json=name2id_json, out_dir=out_dir,
            lenient_substr=lenient_substr,
        )
        rows.append(pd.read_csv(paths["summary"]).iloc[0].to_dict())
    policy = "lenient-substr" if lenient_substr else "strict"
    all_df = pd.DataFrame(rows)
    all_df.to_csv(Path(out_dir)/f"_ALL_entities_recog-nolabel_normid-abbr_{policy}_summary.csv",
                  index=False, encoding="utf-8-sig")
    return all_df


def evaluate_entities_with_label(
    *, model_name: str, models_dir: Path, match_csv: Path,
    gold_jsonl: Path, name2id_json: Path, out_dir: Path,
    lenient_substr: bool = True,
    allowed_labels: Optional[List[str]] = HALD_LABELS
) -> Dict[str, Path]:
    """
    带类型评估：整体 + 每篇文章按类别、全局按类别
    输出：
      per-article（含 detail_by_label JSON）
      summary（整体微/宏）
      summary_by_label（每个类别的累计指标）
      FN/FP 明细（带类型）
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # 映射表（sample_idx -> gold id/pmid/title）
    match_df = pd.read_csv(match_csv, dtype=str)
    m_map = (match_df.query("model == @model_name")
             .dropna(subset=["id"])
             .assign(sample_idx=lambda d: d["sample_idx"].astype(int))
             .set_index("sample_idx"))

    # gold: id -> row
    gold_rows = {}
    with open(gold_jsonl, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                gold_rows[str(obj.get("id"))] = obj
            except json.JSONDecodeError:
                pass

    # 模型数据
    mf = Path(models_dir) / f"{model_name}.json"
    if not mf.exists(): mf = Path(models_dir) / f"{model_name}.jsonl"
    samples = load_json_or_jsonl(mf)

    name2id = load_name2id_map(name2id_json)

    per_rows, rows_fn, rows_fp = [], [], []
    # 为 “summary_by_label” 累计
    label_counters = {lab: {"tp":0, "fp":0, "fn":0, "pred":0, "gold":0} for lab in allowed_labels}

    for idx, sample in enumerate(samples):
        if idx not in m_map.index:
            pred_items = model_entities_with_label(sample, name2id, allowed_labels)
            detail = []
            for lab in allowed_labels:
                pe = [x for x in pred_items if x["type"] == lab]
                label_counters[lab]["pred"] += len(pe)
                detail.append({"label": lab, "tp":0,"fp":len(pe),"fn":None,"precision":0.0,"recall":0.0,"f1":0.0,
                               "pred_cnt":len(pe), "gold_cnt":None})
            per_rows.append({"model": model_name, "sample_idx": idx, "id": None, "pmid": None, "title": None,
                             "pred_cnt": len(pred_items), "gold_cnt": None,
                             "tp": 0, "fp": None, "fn": None,
                             "precision": 0.0, "recall": 0.0, "f1": 0.0,
                             "detail_by_label": json.dumps(detail, ensure_ascii=False)})
            continue

        rid = str(m_map.loc[idx, "id"]); pmid = m_map.loc[idx].get("pmid"); title = m_map.loc[idx].get("title")
        gold_obj = gold_rows.get(rid)
        pred_items = model_entities_with_label(sample, name2id, allowed_labels)

        if gold_obj is None:
            tp, fp, fn = 0, len(pred_items), 0
            p, r, f = prf(tp, fp, fn)
            # 计入类别计数
            for lab in allowed_labels:
                lab_pred = [x for x in pred_items if x["type"] == lab]
                label_counters[lab]["pred"] += len(lab_pred)
                label_counters[lab]["fp"]   += len(lab_pred)
            per_rows.append({"model": model_name, "sample_idx": idx, "id": rid, "pmid": pmid, "title": title,
                             "pred_cnt": len(pred_items), "gold_cnt": 0,
                             "tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f,
                             "detail_by_label": json.dumps([], ensure_ascii=False)})
            for it in pred_items:
                rows_fp.append({"model": model_name, "sample_idx": idx, "id": rid, "pmid": pmid, "title": title,
                                "where": "model_only(FP)", **it})
            continue

        gold_items = gold_entities_with_label(gold_obj, name2id, allowed_labels)
        tp, fp, fn, gold_only, model_only = count_matches_with_label(pred_items, gold_items, lenient_substr)
        p, r, f = prf(tp, fp, fn)

        # 按类别细分
        label_detail = []
        for lab in allowed_labels:
            pe = [x for x in pred_items if x["type"] == lab]
            ge = [x for x in gold_items if x["type"] == lab]
            ttp, tfp, tfn, _, _ = count_matches_with_label(pe, ge, lenient_substr)
            lp, lr, lf = prf(ttp, tfp, tfn)
            label_detail.append({"label": lab, "tp": ttp, "fp": tfp, "fn": tfn,
                                 "precision": lp, "recall": lr, "f1": lf,
                                 "pred_cnt": len(pe), "gold_cnt": len(ge)})
            # 累计到 summary_by_label
            label_counters[lab]["tp"]   += ttp
            label_counters[lab]["fp"]   += tfp
            label_counters[lab]["fn"]   += tfn
            label_counters[lab]["pred"] += len(pe)
            label_counters[lab]["gold"] += len(ge)

        per_rows.append({"model": model_name, "sample_idx": idx, "id": rid, "pmid": pmid, "title": title,
                         "pred_cnt": len(pred_items), "gold_cnt": len(gold_items),
                         "tp": tp, "fp": fp, "fn": fn,
                         "precision": p, "recall": r, "f1": f,
                         "detail_by_label": json.dumps(label_detail, ensure_ascii=False)})

        # 记录 FN/FP 明细
        gb = {(r["key"], r["type"]): r for r in gold_items}
        pb = {(r["key"], r["type"]): r for r in pred_items}
        for pair in sorted(gold_only):
            it = gb[pair]
            rows_fn.append({"model": model_name, "sample_idx": idx, "id": rid, "pmid": pmid, "title": title,
                            "where": "gold_only(FN)", **it})
        for pair in sorted(model_only):
            it = pb[pair]
            rows_fp.append({"model": model_name, "sample_idx": idx, "id": rid, "pmid": pmid, "title": title,
                            "where": "model_only(FP)", **it})

    policy = "lenient-substr" if lenient_substr else "strict"
    per_df = pd.DataFrame(per_rows)
    per_path = Path(out_dir) / build_filename(model_name, "entities_recog", "withlabel", "normid-abbr", policy, "per-article")
    per_df.to_csv(per_path, index=False, encoding="utf-8-sig")

    # —— 总体汇总
    tp_sum = per_df["tp"].fillna(0).astype(int).sum()
    fp_sum = per_df["fp"].fillna(0).astype(int).sum()
    fn_sum = per_df["fn"].fillna(0).astype(int).sum()
    p_micro, r_micro, f_micro = prf(tp_sum, fp_sum, fn_sum)
    p_macro = per_df["precision"].mean()
    r_macro = per_df["recall"].mean()
    f_macro = per_df["f1"].mean()
    sum_df = pd.DataFrame([{
        "model": model_name,
        "articles_evaluated": int(per_df["id"].notna().sum()),
        "tp_total": int(tp_sum), "fp_total": int(fp_sum), "fn_total": int(fn_sum),
        "precision_micro": p_micro, "recall_micro": r_micro, "f1_micro": f_micro,
        "precision_macro": p_macro, "recall_macro": r_macro, "f1_macro": f_macro,
        "norm_strategy": "normid-abbr", "match_policy": policy, "label_mode": "withlabel",
    }])
    sum_path = Path(out_dir) / build_filename(model_name, "entities_recog", "withlabel", "normid-abbr", policy, "summary")
    sum_df.to_csv(sum_path, index=False, encoding="utf-8-sig")

    # —— 每类别汇总
    rows_lab = []
    for lab, c in label_counters.items():
        lp, lr, lf = prf(c["tp"], c["fp"], c["fn"])
        rows_lab.append({"label": lab, "tp": c["tp"], "fp": c["fp"], "fn": c["fn"],
                         "pred_total": c["pred"], "gold_total": c["gold"],
                         "precision": lp, "recall": lr, "f1": lf})
    lbl_df = pd.DataFrame(rows_lab)
    lbl_path = Path(out_dir) / build_filename(model_name, "entities_recog", "withlabel", "normid-abbr", policy, "summary_by_label")
    lbl_df.to_csv(lbl_path, index=False, encoding="utf-8-sig")

    # —— 明细
    fn_df = pd.DataFrame(rows_fn); fp_df = pd.DataFrame(rows_fp)
    fn_path = Path(out_dir) / build_filename(model_name, "entities_recog","withlabel","normid-abbr", policy, "FN")
    fp_path = Path(out_dir) / build_filename(model_name, "entities_recog","withlabel","normid-abbr", policy, "FP")
    fn_df.to_csv(fn_path, index=False, encoding="utf-8-sig")
    fp_df.to_csv(fp_path, index=False, encoding="utf-8-sig")

    return {"per_article": per_path, "summary": sum_path, "summary_by_label": lbl_path, "fn": fn_path, "fp": fp_path}

def batch_evaluate_entities_with_label(
    *, models_dir: Path, match_csv: Path, gold_jsonl: Path,
    name2id_json: Path, out_dir: Path, model_names: Optional[List[str]] = None,
    lenient_substr: bool = True, allowed_labels: Optional[List[str]] = HALD_LABELS
) -> pd.DataFrame:
    if model_names is None:
        names = set()
        for p in Path(models_dir).glob("*.json"): names.add(p.stem)
        for p in Path(models_dir).glob("*.jsonl"): names.add(p.stem)
        model_names = sorted(names)

    rows = []
    for mn in model_names:
        paths = evaluate_entities_with_label(
            model_name=mn, models_dir=models_dir, match_csv=match_csv,
            gold_jsonl=gold_jsonl, name2id_json=name2id_json, out_dir=out_dir,
            lenient_substr=lenient_substr, allowed_labels=allowed_labels
        )
        rows.append(pd.read_csv(paths["summary"]).iloc[0].to_dict())

    policy = "lenient-substr" if lenient_substr else "strict"
    all_df = pd.DataFrame(rows)
    all_df.to_csv(Path(out_dir)/f"_ALL_entities_recog-withlabel_normid-abbr_{policy}_summary.csv",
                  index=False, encoding="utf-8-sig")
    return all_df
