#! /usr/bin/env python
"""
批量调用 DeepSeek / OpenAI, 带断点续跑
"""
from __future__ import annotations
import os, typer
from pathlib import Path
from dotenv import load_dotenv
from haldxai.batch.run_llm_batch import run_batches      # 之前写好的 util
from haldxai.init.config_utils   import load_config

app = typer.Typer(help="执行 LLM 推理 (jsonl → result_*.jsonl)")

@app.command()
def run(
    task_name: str = typer.Option(..., "--task"),
    years: str     = typer.Option("",  "--years", help="逗号分隔，如 2023,2024"),
    project_root: Path = typer.Option(..., "--root"),
):
    # 1. API key / base_url
    load_dotenv(project_root / ".env", override=False)
    cfg = load_config(project_root / "configs" / "config.yaml")
    api_key  = os.getenv("DEEPSEEK_API_KEY")
    api_base = cfg["api"]["deepseek"]["base_url"]

    yrs = [int(y) for y in years.split(",") if y] or None

    in_dir  = project_root / f"data/interim/batch_process/batch_llm_ner_input/{task_name}"
    out_dir = project_root / f"data/interim/batch_process/batch_results/{task_name}"
    run_batches(in_dir, out_dir, years=yrs, api_key=api_key, api_base=api_base)

if __name__ == "__main__":
    app()
