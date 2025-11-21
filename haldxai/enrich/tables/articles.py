#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_articles_table.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
重新生成 `articles.csv`，同时把 `pub_date` 统一规范为
**YYYY‑MM‑DD** 格式（缺月/日时补 `01`）。

步骤
====
1. `pmid` 仍按原逻辑处理成字符串。
2. `pub_date` 清洗：
   * `YYYY-MM`      -> `YYYY-MM-01`
   * `YYYY`         -> `YYYY-01-01`
   * 已是 `YYYY-MM-DD` 保留。
   * 解析失败则置为空串。
3. 保存为 UTF‑8，路径 `data/database/articles.csv`。

用法
----
from pathlib import Path
import pandas as pd
from haldxai.workflow.build_articles_table import build_articles

df = pd.read_parquet("raw_articles.parquet")
build_articles(Path("."), df, force=True)
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import re

__all__ = ["build_articles"]

_DATE_RE = re.compile(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$")


def _normalize_date(date_str: str | float | int | None) -> str:
    """
    把各种长度的日期字符串归一化为 YYYY-MM-DD
    • YYYY-MM → YYYY-MM-01
    • YYYY    → YYYY-01-01
    超出范围或解析失败返回空串
    """
    if date_str is None or pd.isna(date_str):
        return ""

    s = str(date_str).strip()
    m = _DATE_RE.fullmatch(s)
    if not m:
        return ""

    year, month, day = m.groups()
    month = month or "01"
    day   = day   or "01"

    # ---- 范围校验 ----
    y, m_, d_ = int(year), int(month), int(day)
    if not (1 <= m_ <= 12 and 1 <= d_ <= 31):
        return ""                    # 非法日期直接丢弃 / 留空

    return f"{y:04d}-{m_:02d}-{d_:02d}"


def build_articles(
    project_root: Path,
    df_articles: pd.DataFrame,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """清洗并导出 *articles.csv*。"""

    out_dir = project_root / "data/database"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "articles.csv"

    if out_csv.exists() and not force:
        print("🟡  articles.csv 已存在，跳过。使用 force=True 可覆盖重新生成。")
        return pd.read_csv(out_csv)

    print("▶  构建 articles.csv …")

    # --- PMID 处理 ---------------------------------------------------------
    df_articles = df_articles.copy()
    df_articles["pmid"] = (
        pd.to_numeric(df_articles["pmid"], errors="coerce")
        .fillna(0)
        .astype("Int64")
        .astype(str)
        .replace({"<NA>": ""})
    )

    # --- pub_date 规范化 ---------------------------------------------------
    df_articles["pub_date"] = df_articles["pub_date"].apply(_normalize_date)

    # --- 导出 --------------------------------------------------------------
    df_articles.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"✓ articles.csv 写出 {len(df_articles):,} 行 -> {out_csv}")
    return df_articles
