#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
清洗 PubMed 汇总表 → 标注影响因子 → 按年份拆分

CLI
----
python -m haldxai.pubmed.run_postprocess_articles \
    --task aging-related      # 任务名，对应 articles_summary_{task}.csv
    --force                   # 已存在年度 CSV 时仍覆盖

Notebook
--------
from haldxai.pubmed.run_postprocess_articles import run
run(task="aging-related")
"""
from __future__ import annotations

import sys, argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# 项目根
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# 内部工具
from haldxai.init.config_utils import load_config, update_config
from haldxai.pubmed.clean  import save_yearly_data            # 你已有的 util
from haldxai.pubmed.impact import annotate_journals_with_if   # 你已有的 util

# ------------------------------------------------------------
# 主函数
# ------------------------------------------------------------
def run(task: str = "aging-related", force: bool = False) -> None:
    """清洗 & 拆分."""
    # 0. 读取配置 / 环境
    cfg_path = project_root / "configs" / "config.yaml"
    env_path = project_root / ".env"
    cfg = load_config(cfg_path)
    load_dotenv(env_path, override=False)

    # 1. 关键路径
    summary_csv = Path(cfg[f"articles_summary_{task}"])
    interim_root = project_root / cfg.get("intermediate_dir", "data/interim")
    data_dir    = interim_root / "articles" / f"articles_{task}"
    data_dir.mkdir(parents=True, exist_ok=True)               # double-check

    summary_if = data_dir / f"articles_summary_{task}_with_if.csv"

    # 2. 读取 & 清洗
    if not summary_csv.exists():
        raise FileNotFoundError(f"{summary_csv} 不存在，请先抓取原始文献。")

    df = pd.read_csv(summary_csv)
    print(f"📥 原始行数: {len(df)}")

    df = (df.dropna(subset=["pmid", "title", "abstract",
                            "pub_date", "journal_full_title"])
            .query("not pub_types.str.contains('Retracted Publication', na=False)",
                   engine='python')
            .drop_duplicates(subset="pmid")
           )
    df["pmid"] = df["pmid"].astype(str)
    df.reset_index(drop=True, inplace=True)
    print(f"🧹 清洗后行数: {len(df)}")

    # 3. 影响因子标注
    df = annotate_journals_with_if(df)
    print(f"⭐ 影响因子完成，剩余行数: {len(df)}")

    # 4. 保存总表
    summary_if.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(summary_if, index=False, encoding="utf-8-sig")
    print(f"✅ 保存汇总 → {summary_if}")

    # 5. 按年份拆分
    print("📂 正在生成年度 CSV ...")
    yearly_dir = summary_if.with_suffix("")        # same 前缀
    year_df = save_yearly_data(df, str(yearly_dir))

    print("\n📊 年度文献量：")
    print(year_df.groupby("year").size().reset_index(name="count").to_string(index=False))

    # 6. 更新配置
    update_config(cfg_path, {
        f"articles_summary_{task}_with_if_dir": str(data_dir),
        f"articles_summary_{task}_with_if": str(summary_if),
        "last_clean_date": datetime.now().strftime("%Y-%m-%d"),
    })

    print("🎉 后处理完成，config.yaml 已更新")

# ------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--task",  default="aging-related")
    pa.add_argument("--force", action="store_true",
                    help="已存在年度 CSV 时仍覆盖")
    args = pa.parse_args()
    run(task=args.task, force=args.force)
