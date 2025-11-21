#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量抓取 PubMed 文献 & 更新配置
--------------------------------
CLI:
    python -m haldxai.pubmed.run_fetch_articles \
        --task aging-related        # 对应配置里的 task 名
        --start_year 1950           # 可选：覆盖 config.yaml 中 last_update_year
        --end_year 2025           # 可选：覆盖当前最新日期
        --retmax 1000          # 可选：覆盖默认的 retmax
        --batch_size 200           # 可选：覆盖默认的 batch_size
Notebook:
    from haldxai.pubmed.run_fetch_articles import run
    run(task="aging-related")
"""
from __future__ import annotations

import os, sys, yaml, argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────
# ✨ 项目根目录 & 依赖
# ──────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from haldxai.init.config_utils import (
    load_config, save_config, update_config, show_config
)
from haldxai.pubmed.fetch   import fetch_pubmed_data, generate_query_with_time
from haldxai.pubmed.process import generate_monthly_ranges


# ──────────────────────────────────────────────────────
# 🔑 1. 读取 config & .env
# ──────────────────────────────────────────────────────
cfg_path = project_root / "configs" / "config.yaml"
env_path = project_root / ".env"

cfg = load_config(cfg_path)
load_dotenv(env_path, override=False)          # 环境变量优先来自系统环境


# ──────────────────────────────────────────────────────
# 🏃 主逻辑
# ──────────────────────────────────────────────────────
def run(
    task: str = "aging-related",
    start_year: int | None = None,
    end_year: int | None = None,
    retmax: int | None = None,
    batch_size: int | None = None
) -> None:
    """
    Parameters
    ----------
    task : str
        任务名称（将写入 articles/articles_{task} 下）
    start_year : int | None
        不指定则用 config['last_update_year']，否则覆盖
    end_year : int | None
        不指定则用现在的最新日期，否则覆盖
    retmax : int | None
        PubMed 查询的最大返回数量，不传则用 config.retmax
    batch_size : int | None
        PubMed fetch 的批次大小，不传沿用 config.batch.chunk_size
    """

    # ========== 1. 输入参数与配置 ==========
    task_key_dir  = f"articles_info_{task}_dir"
    task_key_sum  = f"articles_summary_{task}"
    email  = os.getenv("PUBMED_EMAIL")      or cfg.get("api", {}).get("pubmed", {}).get("email")
    api_key = os.getenv("PUBMED_API_KEY")   or cfg.get("api", {}).get("pubmed", {}).get("api_key")

    if not email or not api_key:
        raise RuntimeError("❌ PUBMED_EMAIL / PUBMED_API_KEY 未设置，请写入 .env 或 config.yaml")

    # PubMed 查询语句可放到 configs，但这里写死与之前保持一致
    query_core = cfg.get("pubmed_query", {}).get(task)

    # ========== 2. 路径准备 ==========
    data_dir = project_root / cfg["data_dir"] / "raw" /"articles" / f"articles_{task}"
    data_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = data_dir / f"articles_summary_{task}.csv"

    # ========== 3. 抓取范围 ==========
    start_year = start_year or cfg.get("last_update_year", 1945)
    end_year   = end_year or datetime.now().year
    retmax = retmax or cfg.get("retmax", None)
    batch_size = batch_size or cfg.get("batch", {}).get("chunk_size", 200)

    # ========== 4. 抓取 ==========
    for start_date, end_date in generate_monthly_ranges(start_year, end_year):
        print(f"🔍 查询 {start_date} ~ {end_date}")
        query = generate_query_with_time(query_core, start_date, end_date)
        df = fetch_pubmed_data(
            query=query,
            email=email,
            summary_file=str(summary_csv),
            api_key=api_key,
            retmax=retmax,
            batch_size=batch_size
        )

    # ========== 6. 更新配置 ==========
    final_cfg = {
        task_key_dir : str(data_dir),
        task_key_sum : str(summary_csv),
        "last_update_date" : datetime.now().strftime("%Y-%m-%d"),
    }
    if "pub_date" in df.columns:
        pub_years = df["pub_date"].dropna().str.split("-").str[0].astype(int)
        if not pub_years.empty:
            final_cfg["last_update_year"] = int(pub_years.max())

    update_config(cfg_path, final_cfg)
    print("🎉 抓取任务完成，config.yaml 已更新")


# ──────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--task",        default="aging-related")
    pa.add_argument("--start_year",  type=int, help="覆盖 last_update_year")
    pa.add_argument("--end_year", type=int, help="覆盖当前最新日期")
    pa.add_argument("--retmax", type=int, help="PubMed 查询的最大返回数量，不传则用 config.retmax")
    pa.add_argument("--batch_size",  type=int, help="覆盖默认 batch_size")
    args = pa.parse_args()

    run(
        task        = args.task,
        start_year  = args.start_year,
        end_year    = args.end_year,
        retmax      = args.retmax,
        batch_size  = args.batch_size
    )
