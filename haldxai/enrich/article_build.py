#!/usr/bin/env python
# build_articles_std.py
# ------------------------------------------------------------
# ⬇️ 构建 HALD 里的 ARTICLE 表 + 缓存 parquet
#   · 输入 : data/raw/articles/articles_aging-related/articles_summary_aging-related.csv
#   · 输出 : data/hald_database/ARTICLE.csv
#           cache/articles.parquet
# ------------------------------------------------------------
from __future__ import annotations
import logging, json, csv, joblib
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(levelname)s: %(message)s", level=logging.INFO)

# ------------------------------------------------------------
# 1.  核心函数
# ------------------------------------------------------------
def build_articles(
    project_root : Path,
    *,
    force       : bool = False,
    model_path  : Path | None = None,
) -> None:
    """
    读取整合后的文献 CSV ➜ 预测 aging_prob ➜ 输出 ARTICLE.csv / articles.parquet
    -------------------------------------------------------------------------
    参数
    ----
    project_root : 项目根目录 Path（外层 CLI 会把 str -> Path 处理）
    force        : 输出已存在时是否覆盖
    model_path   : 自定义 TF-IDF + Logistic/Trees… 模型路径。
                   若 None，则默认 `<project_root>/model/age-related/tfidf-clf_aging_classifier.pkl`
    """
    # ---------- 目录 ----------
    proj = project_root
    art_csv = proj / "data/interim/articles/articles_aging-related" / "articles_summary_aging-related_with_if.csv"
    cache   = proj / "cache"
    cache.mkdir(parents=True, exist_ok=True)

    out_pq   = cache   / "articles.parquet"

    if not force and out_pq.exists():
        logger.info("ARTICLE 已存在，跳过。传入 force=True 可重新生成。")
        return

    # ---------- 读取 ----------
    logger.info("▶ 读取文献 CSV …")
    df = pd.read_csv(art_csv, low_memory=False)
    logger.info(f"   文献数量: {len(df):,}")

    # ---------- 选列 + 清洗 ----------
    keep_cols = [
        'pmid', 'title', 'abstract', 'pub_date', 'authors', 'pub_types', 'journal',
        'journal_full_title', 'journal_abbr', 'jcr', 'factor', 'issn', 'nlm_id', 'eissn'
    ]
    df = df[keep_cols].copy()
    df = df[df["abstract"].notna() & (df["abstract"].str.strip() != "")]
    logger.info(f"   有效摘要记录: {len(df):,}")

    # ---------- 载入模型 ----------
    if model_path is None:
        model_path = proj / "models/aging_classifier_tfidf_lr_v1/model.pkl"
    logger.info(f"▶ 加载模型: {model_path}")
    clf = joblib.load(model_path)

    # ---------- 预测概率 ----------
    logger.info("▶ 预测 aging_prob …")
    df["aging_prob"] = clf.predict_proba(df["abstract"])[:, 1]

    # ---------- 保存 ----------
    df.to_parquet(out_pq, index=False)
    logger.info("\n🎉 ARTICLE 构建完成")
    logger.info(f"   • {out_pq}  ({len(df):,} 行)")

# ------------------------------------------------------------
# 2.  Typer CLI 入口（可直接 python build_articles_std.py run …）
# ------------------------------------------------------------
if __name__ == "__main__":
    import typer, rich
    app = typer.Typer(pretty_exceptions_show_locals=False)

    @app.command("run")
    def _run(
        root : str = typer.Option(..., help="项目根目录"),
        model: str = typer.Option(None, help="自定义模型 pkl 路径"),
        force: bool= typer.Option(False, "--force", "-f", help="覆盖已存在输出")
    ):
        build_articles(Path(root), force=force, model_path=(Path(model) if model else None))

    rich.print("[bold green]HALDxAI[/]  · 构建 ARTICLE 表")
    app()
