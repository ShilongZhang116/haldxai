"""
批量实体识别脚本
用法（CLI）:
    python -m haldxai.ner.run_spacy_ner \
        --models en_ner_bionlp13cg_md en_ner_bc5cdr_md \
        --input_dir data/articles_info \
        --prefix hald_literature_with_if_total \
        --output_dir data/ner_output/spacy
"""

from __future__ import annotations
from pathlib import Path
import argparse
import pandas as pd
from tqdm.auto import tqdm

from haldxai.ner.utils import detect_years, build_save_path, ensure_spacy_model

from pathlib import Path, PurePath
import sys, yaml

PROJECT_ROOT = Path().resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"
CONFIG = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))

# -------- 主函数 -------- #
def batch_ner_for_year(
    year: int,
    model: str,
    input_dir: Path,
    prefix: str,
    output_dir: Path,
) -> None:
    save_path = build_save_path(output_dir, model, year)
    if save_path.exists():
        print(f"⏭️ {save_path.name} 已存在，跳过")
        return

    file_path = input_dir / f"{prefix}_Y{year}.csv"
    if not file_path.exists():
        print(f"⚠️ 未找到 {file_path}，跳过")
        return

    print(f"🔍 {year} | {model}")

    nlp = ensure_spacy_model(model, local_repo=PROJECT_ROOT / "models/SciSpacy")

    df = pd.read_csv(file_path)
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{model} {year}"):
        doc = nlp(str(row.get("abstract", "")))
        pmid = str(row.get("pmid", ""))
        for ent in doc.ents:
            results.append(
                dict(
                    pmid=pmid,
                    entity_text=ent.text,
                    label=ent.label_,
                    year=year,
                    model=model,
                    source="abstract",
                )
            )
    pd.DataFrame(results).to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"✅ 保存 {save_path}")

# -------- CLI -------- #
def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--input_dir", type=Path, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--years", nargs="*", type=int, help="指定年份列表，可选")
    args = parser.parse_args()

    years = args.years or detect_years(args.input_dir, args.prefix)
    print(f"📅 待处理年份：{years}")

    for model in args.models:
        for y in years:
            batch_ner_for_year(
                year=y,
                model=model,
                input_dir=args.input_dir,
                prefix=args.prefix,
                output_dir=args.output_dir,
            )

if __name__ == "__main__":
    cli()
