#! /usr/bin/env python
"""
* 把带 IF 的摘要表拆分为年度批次
* Filter + Prompt 封装
"""
from __future__ import annotations
from pathlib import Path
import yaml, textwrap, typer

from haldxai.batch.split         import split_year_to_batches
from haldxai.batch.build_llm_input import csv_to_jsonl
from haldxai.ner.utils           import detect_years
from haldxai.init.config_utils   import load_config

app = typer.Typer(help="准备 LLM NER 输入文件 (csv → jsonl)")

def _load_prompts(root: Path):
    prm = yaml.safe_load((root / "configs" / "prompts.yaml").read_text(encoding="utf-8"))
    return textwrap.dedent(prm["system_prompt"])

def _iter_tasks(root: Path):
    cfg = yaml.safe_load((root / "configs" / "llm_tasks.yaml").read_text())
    for name, task in cfg.items():
        yield name, task

@app.command()
def run(
    project_root: Path = typer.Option(
        ..., "--root", help="HALDxAI-Project 根目录"
    ),
    force: bool = typer.Option(
        False, "--force", help="已存在 *.jsonl 时是否覆盖"
    ),
):
    system_prompt = _load_prompts(project_root)

    cfg = load_config(project_root / "configs" / "config.yaml")
    art_dir = Path(cfg["articles_summary_aging-related_with_if_dir"])
    prefix  = "articles_summary_aging-related_with_if"

    for task_name, tcfg in _iter_tasks(project_root):
        year_list = (detect_years(art_dir, prefix)
                     if tcfg["years"] == "auto"
                     else [int(y) for y in tcfg["years"]])
        print(f"🚀 {task_name} | years = {year_list}")

        out_csv  = project_root / "data/interim/batch_process/batch_articles_info" / task_name
        out_json = project_root / "data/interim/batch_process/batch_llm_ner_input" / task_name
        out_json.mkdir(parents=True, exist_ok=True)

        # ① 年→批 CSV
        for yr in year_list:
            split_year_to_batches(
                year=yr, task_name=task_name,
                input_dir=art_dir, prefix=prefix,
                output_dir=out_csv, batch_size=tcfg["batch_size"],
                filter_method=tcfg["filter_method"]
            )

        # ② CSV → jsonl
        for csv_f in out_csv.glob("*.csv"):
            js = out_json / f"{csv_f.stem}.jsonl"
            if js.exists() and not force:
                continue
            csv_to_jsonl(csv_f, js, system_prompt, tcfg["model_name"])

    print("🎉 prepare_llm_batches 完成")

if __name__ == "__main__":
    app()
