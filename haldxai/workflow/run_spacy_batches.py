#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量调用 spaCy BioNER → 保存到 data/interim/ner_output/spacy

CLI
---
python -m haldxai.workflow.run_spacy_batches \
       --models en_ner_bc5cdr_md en_ner_bionlp13cg_md \
       --years 2024 2025                       # 或 --years auto
       --task aging-related                    # 默认
       --root  F:/Project/HALDxAI-Project      # 可选，默认脚本所在 repo 根
"""

from __future__ import annotations
import sys, yaml, typer
from pathlib import Path
from typing import List

# ── 项目根 & 内部 import ─────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))

from haldxai.init.config_utils import load_config
from haldxai.ner.run_spacy_ner import batch_ner_for_year
from haldxai.ner.utils        import detect_years

# Typer CLI
app = typer.Typer(add_completion=False)

# -------------------------------------------------------
def _resolve_years(years_arg: List[str] | None,
                   art_dir: Path, prefix: str) -> List[int]:
    """years_arg=['auto'] 时自动检测；否则转 int"""
    if not years_arg or years_arg == ["auto"]:
        return detect_years(art_dir, prefix)
    return [int(y) for y in years_arg]

# -------------------------------------------------------
def run(task: str = "aging-related",
        models: List[str] | None = None,
        years:  List[int] | None = None,
        root:   Path | str = ROOT) -> None:
    """
    参数
    ----
    task   : config 中的任务名前缀 (articles_summary_{task}_with_if)
    models : spaCy 模型列表；None=全部三个
    years  : 年度列表；None=自动检测；["auto"]=自动检测
    root   : 项目根目录
    """
    root = Path(root)
    cfg  = load_config(root / "configs" / "config.yaml")

    art_dir   = Path(cfg[f"articles_summary_{task}_with_if_dir"])
    prefix    = f"articles_summary_{task}_with_if"
    out_dir   = root / "data" / "interim" / "ner_output" / "spacy"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = models or ["en_ner_bc5cdr_md",
                        "en_ner_bionlp13cg_md",
                        "en_ner_jnlpba_md"]
    years  = _resolve_years(years or ["auto"], art_dir, prefix)

    typer.echo(f"🏷  Task={task}  Years={years}  Models={models}")

    for m in models:
        for y in years:
            typer.echo(f"🚀 {m} @ {y}")
            batch_ner_for_year(
                year        = y,
                model       = m,
                input_dir   = art_dir,
                prefix      = prefix,
                output_dir  = out_dir,
            )
    typer.echo("🎉 spaCy NER 批处理完成")

# -------------------------------------------------------
@app.command()
def cli(models: List[str] = typer.Option(None, help="spaCy 模型名列表"),
        years:  List[str] = typer.Option(["auto"],
                          help="年份列表，如 2023 2024，或 auto"),
        task:   str = typer.Option("aging-related", help="任务名前缀"),
        root:   Path = typer.Option(ROOT, help="项目根目录")):
    """命令行入口"""
    run(task=task, models=models, years=years, root=root)

# python -m haldxai.workflow.run_spacy_batches ...
if __name__ == "__main__":
    app()
