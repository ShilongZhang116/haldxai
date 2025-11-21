#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""validate_utils.py
~~~~~~~~~~~~~~~~~~~~
一个既可 **import**，也可 **CLI** 运行的工具模块，帮助你：

1. **validate_graph** — 校验 *nodes.csv* / *relationships.csv* 是否配套；
2. **clean_relationships** — 找到悬空关系后，写出一个“已清理”的关系 CSV；

用法示例
---------
➊ 交互式检查
```
from validate_utils import validate_graph, clean_relationships
ok, miss_df = validate_graph("nodes.csv", "rels.csv")
if not ok:
    clean_relationships("nodes.csv", "rels.csv", "rels_clean.csv")
```

➋ CLI 一键修复
```
python -m validate_utils \
   --nodes data/database/nodes.csv \
   --rels  data/database/relationships.csv \
   --fix   data/database/relationships_clean.csv
```
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd

__all__ = [
    "validate_graph",
    "clean_relationships",
]

# ---------------------------------------------------------------------------
# 核心校验函数
# ---------------------------------------------------------------------------

def _load_node_set(nodes_path: Path, id_col: str) -> set[str]:
    """读取 nodes.csv 中的 ID 集合 (str)。"""
    return set(
        pd.read_csv(nodes_path, usecols=[id_col], dtype=str)[id_col].astype(str)
    )


def _load_rels(rels_path: Path, start_col: str, end_col: str) -> pd.DataFrame:
    """读取 relationships.csv 指定两列并转 str。"""
    return pd.read_csv(rels_path, usecols=[start_col, end_col], dtype=str)


def validate_graph(
    project_root: str | Path,
    nodes_path: str | Path,
    rels_path: str | Path,
    id_col: str = "node_id:ID",
    start_col: str = ":START_ID",
    end_col: str = ":END_ID",
    sample: int = 5,
) -> Tuple[bool, pd.DataFrame]:
    """检查关系文件是否引用了缺失节点。

    返回 *(is_ok, missing_df)*，其中 `missing_df` 列为 [`role`, `missing_id`].
    """
    nodes_path, rels_path = Path(project_root / nodes_path), Path(project_root / rels_path)

    node_set = _load_node_set(nodes_path, id_col)
    rels_df  = _load_rels(rels_path, start_col, end_col)

    missing_start = rels_df.loc[~rels_df[start_col].isin(node_set), start_col]
    missing_end   = rels_df.loc[~rels_df[end_col].isin(node_set), end_col]

    missing_df = pd.concat([
        pd.DataFrame({"role": "start", "missing_id": missing_start}),
        pd.DataFrame({"role": "end",   "missing_id": missing_end}),
    ]).drop_duplicates()

    is_ok = missing_df.empty

    if is_ok:
        print("✅ 校验通过：relationships.csv 的所有节点均存在于 nodes.csv。")
    else:
        print(f"❌ 检测到缺失节点：{len(missing_df)} 条")
        if sample > 0:
            print(missing_df.head(sample))

    return is_ok, missing_df


# ---------------------------------------------------------------------------
# 清理函数
# ---------------------------------------------------------------------------

def clean_relationships(
    project_root: str | Path,
    nodes_path: str | Path,
    rels_path: str | Path,
    output_path: str | Path,
    id_col: str = "node_id:ID",
    start_col: str = ":START_ID",
    end_col: str = ":END_ID",
) -> None:
    """生成一个已移除悬空关系的新 CSV。

    - 仅当关系两端节点 *都* 在节点文件中出现时保留。
    - 其余列原样保留。
    """
    nodes_path, rels_path, output_path = map(Path, (project_root / nodes_path, project_root / rels_path, project_root / output_path))

    node_set = _load_node_set(nodes_path, id_col)
    rels_df  = pd.read_csv(rels_path, dtype=str)

    mask_start = rels_df[start_col].isin(node_set)
    mask_end   = rels_df[end_col].isin(node_set)
    cleaned_df = rels_df[mask_start & mask_end]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(output_path, index=False, encoding="utf-8")

    removed = len(rels_df) - len(cleaned_df)
    print(f"🧹 已写出 cleaned CSV → {output_path} (移除 {removed} 条悬空关系)")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="校验并可清理 relationships.csv 中的悬空节点")
    p.add_argument("--nodes", required=True, type=Path, help="nodes.csv 路径")
    p.add_argument("--rels",  required=True, type=Path, help="relationships.csv 路径")
    p.add_argument("--sample", type=int, default=5, help="缺失样例打印数量 (<=0 不打印)")
    p.add_argument("--fix",    type=Path, help="输出已清理 CSV 的保存路径 (可覆盖原文件)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:  # noqa: D401
    args = _parse_args(argv)

    ok, missing = validate_graph(args.nodes, args.rels, sample=args.sample)

    if (not ok) and args.fix:
        clean_relationships(args.nodes, args.rels, args.fix)
        ok = True  # 认为已修复

    sys.exit(0 if ok else 1)


if __name__ == "__main__":  # pragma: no cover
    main()
