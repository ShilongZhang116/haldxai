# merge_annotations.py
# ===========================================================
from __future__ import annotations
from pathlib import Path
import pandas as pd, glob, textwrap, csv

def _append_if_exist(df_new: pd.DataFrame, f_out: Path,
                     subset: list[str] | None = None) -> pd.DataFrame:
    """
    若 f_out 已存在 → 追加旧数据并去重
    subset 可选：决定用哪些列判断重复（默认全部列）
    """
    if f_out.exists():
        df_old = pd.read_csv(f_out, dtype=str, low_memory=False)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all = df_all.drop_duplicates(subset=subset or None)
        return df_all
    return df_new

def _read_with_model(collected: list[pd.DataFrame], pattern: str, kind: str):
    """
    把 pattern 匹配到的所有 CSV 读取为 DF，并在每条记录加上 model_name
    kind = 'entities' / 'relationships' ⇒ 决定如何给列起别名
    """
    for fp in sorted(glob.glob(pattern)):
        # 1) 取文件名，例如 annotated_entities_AgingRelated-DeepSeekR1-7B.csv
        fname = Path(fp).stem                      # → annotated_entities_AgingRelated-DeepSeekR1-7B
        parts = fname.split("_", maxsplit=2)

        # 2) model_name = 第 3 段（若文件名中只有两段就取第二段）
        model_name = parts[-1] if len(parts) >= 3 else parts[-1]

        # 3) 读取 CSV
        df = pd.read_csv(fp, dtype=str, low_memory=False)

        # 4) 写入新列
        df.insert(0, "model_name", model_name)

        # ※ 如果你想再加 task_name，可在这里用 parts[1] 提取

        collected.append(df)

    # 全部合并
    return pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()

# -----------------------------------------------------------------
def merge_ann(
    proj: str | Path,
    ner_dir: str | Path = "data/ner_dict",
    out_dir: str | Path = "data/finals",
    ents_name: str      = "all_annotated_entities.csv",
    rels_name: str      = "all_annotated_relationships.csv",
    # ▶ 决定“如何去重”的列（按需调整）
    ents_unique_cols: list[str] | None = None,
    rels_unique_cols: list[str] | None = None,
):
    """
    增量式合并 annotated_entities*.csv / annotated_relationships*.csv
    ---------------------------------------------------------------
    若 out_dir/xxx 已存在，则把“新增”文件 append 并去重后再写回。
    """
    proj     = Path(proj)
    ner_dir  = Path(ner_dir)  if Path(ner_dir).is_absolute() else proj/ner_dir
    out_dir  = Path(out_dir)  if Path(out_dir).is_absolute() else proj/out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    ents_pat = str(ner_dir / "annotated_entities*.csv")
    rels_pat = str(ner_dir / "annotated_relationships*.csv")


    # -------- Entities --------
    df_ents_new = _read_with_model([], ents_pat, "entities")
    if not df_ents_new.empty:
        f_out_ents = out_dir / ents_name
        df_ents_all = _append_if_exist(df_ents_new, f_out_ents, ents_unique_cols)
        df_ents_all.to_csv(f_out_ents, index=False, encoding="utf-8-sig",
                           quoting=csv.QUOTE_MINIMAL)
        print(f"✅ Entities → {f_out_ents}  ({len(df_ents_all):,} rows)")
    else:
        print("⚠️ 未发现 annotated_entities*.csv")

    # -------- Relationships --------
    df_rels_new = _read_with_model([], rels_pat, "relationships")
    if not df_rels_new.empty:
        f_out_rels = out_dir / rels_name
        df_rels_all = _append_if_exist(df_rels_new, f_out_rels, rels_unique_cols)
        df_rels_all.to_csv(f_out_rels, index=False, encoding="utf-8-sig",
                           quoting=csv.QUOTE_MINIMAL)
        print(f"✅ Relationships → {f_out_rels}  ({len(df_rels_all):,} rows)")
    else:
        print("⚠️ 未发现 annotated_relationships*.csv")

# -----------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys
    pa = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""
            增量式合并 data/ner_dict 下的注释文件
            """))
    pa.add_argument("--proj",    required=True, help="项目根目录")
    pa.add_argument("--ner_dir", default="data/ner_dict")
    pa.add_argument("--out_dir", default="data/finals")
    args = pa.parse_args()

    try:
        merge_ann(args.proj, args.ner_dir, args.out_dir)
    except Exception as e:
        print(f"❌ 运行失败: {e}")
        sys.exit(1)
