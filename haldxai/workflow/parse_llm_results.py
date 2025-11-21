#! /usr/bin/env python
"""
result_*.jsonl → Entities_*.csv / Relationships_*.csv
"""
from pathlib import Path
import typer
from haldxai.batch.parse_llm_output import parse_by_year

app = typer.Typer(help="解析 LLM 输出")

@app.command()
def run(
    task   : str        = typer.Option(..., "--task"),
    root   : Path       = typer.Option(..., "--root"),
    years  : str        = typer.Option("", "--years", help="2023,2024"),
):
    yrs = [int(x) for x in years.split(",") if x] or None
    parse_by_year(
        task_name        = task,
        project_root     = root,
        article_info_dir = root / f"data/interim/batch_process/batch_articles_info/{task}",
        batch_input_dir  = root / f"data/interim/batch_process/batch_llm_ner_input/{task}",
        batch_result_dir = root / f"data/interim/batch_process/batch_results/{task}",
        save_dir         = root / f"data/interim/ner_output/{task}",
        years            = yrs,
    )

if __name__ == "__main__":
    app()
