"""
批量调用 OpenAI / DeepSeek Chat 接口，把结果写回 jsonl

CLI 例子：
    python -m haldxai.batch.run_llm_batch \
      --input_dir  data/batch_process/batch_llm_ner_input/JCRQ1-IF10-DeepSeekV3 \
      --output_dir data/batch_process/batch_results/JCRQ1-IF10-DeepSeekV3 \
      --years 2023 2024 2025 \
      --api_key sk-xxxxx

Notebook 例子：
    from haldxai.batch.run_llm_batch import run_batches
    run_batches("in_dir", "out_dir", years=[2023,2024], api_key="sk-xxx")
"""
from __future__ import annotations

from pathlib import Path
from typing   import Iterable, List
import os, re, json, backoff, jsonlines, rich, requests, httpx
from tqdm.notebook import tqdm
from openai import OpenAI                        # 仅需这个，不引入 openai.error

# ────────────────────────── 工具函数 ────────────────────────── #
def _count_lines(fp: Path) -> int:
    with fp.open(encoding="utf-8") as f:
        return sum(1 for _ in f)

def _extract_year(fname: str) -> int | None:
    m = re.search(r"Y(\d{4})", fname)
    return int(m.group(1)) if m else None

# ---------- 指定异常是否可重试 ---------- #
def _retryable(e: Exception) -> bool:
    retry_http_status = {429, 500, 502, 503, 504}
    status = getattr(e, "status_code", None) or getattr(
        getattr(e, "response", None), "status_code", None
    )
    return isinstance(e, (requests.RequestException, httpx.HTTPError)) or status in retry_http_status

# ---------- 判断 result 文件是否完整 ---------- #
def _is_output_complete(fp_in: Path, fp_out: Path) -> bool:
    if not fp_out.exists():
        return False
    try:
        return _count_lines(fp_out) >= _count_lines(fp_in)
    except Exception:
        return False

# ---------- 加载已完成的 request_id ---------- #
def _load_completed_ids(fp_out: Path) -> set[str]:
    if not fp_out.exists():
        return set()

    done: set[str] = set()
    try:
        with jsonlines.open(fp_out) as reader:
            for obj in reader:
                qid = (
                    obj.get("request_id")
                    or obj.get("request", {}).get("request_id")
                )
                if qid:
                    done.add(qid)
    except Exception as e:
        rich.print(f"[red]⚠️ 读取 {fp_out.name} 出错: {e}[/red]")
    return done

# ---------- 调用大模型，带指数退避 ---------- #
@backoff.on_exception(backoff.expo, Exception, max_time=180, giveup=lambda e: not _retryable(e))
def _call_chat(client: OpenAI, req: dict) -> dict:
    rsp = client.chat.completions.create(
        model=req["model"],
        messages=req["messages"],
        stream=False,
    )
    return rsp.model_dump()

# ---------- 处理单个 jsonl 文件 ---------- #
def _process_one_jsonl(fp_in: Path, fp_out: Path, client: OpenAI):
    total = _count_lines(fp_in)
    completed = _load_completed_ids(fp_out)

    ok = fail = 0
    with fp_in.open(encoding="utf-8") as fin, jsonlines.open(fp_out, "a") as fout:
        with tqdm(total=total, initial=len(completed), desc=fp_in.name, unit="req") as bar:
            for line in fin:
                req = json.loads(line)
                qid = req.get("request_id")                                     # ⚠️ 确保构建输入时已写入
                if qid in completed:
                    continue

                try:
                    rsp = _call_chat(client, req)
                    fout.write({"request": req, "response": rsp})
                    ok += 1
                except Exception as e:
                    rich.print(f"[red]✖ {e}[/red]")
                    fail += 1
                bar.update(1)

    rich.print(f"✅ {fp_in.name}: 新增 {ok}, 跳过 {len(completed)}, 失败 {fail}")

# ────────────────────────── 对外主函数 ────────────────────────── #
def run_batches(
    input_dir : Path | str,
    output_dir: Path | str,
    years    : List[int] | None = None,
    api_key  : str | None = None,
    api_base : str | None = None,
):
    """
    Parameters
    ----------
    input_dir :  含 *.jsonl 输入文件的目录
    output_dir:  结果保存目录
    years     :  仅处理这些年份 (文件名里带 Y2023 等)；None 表示全部
    api_key   :  OpenAI / DeepSeek Key；为空则读取环境变量 OPENAI_API_KEY
    api_base  :  Base URL；为空则环境变量 OPENAI_API_BASE 或默认 'https://api.deepseek.com'
    """
    input_dir  = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key  = api_key  or os.getenv("OPENAI_API_KEY")
    api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.deepseek.com")
    if not api_key:
        raise RuntimeError("未提供 api_key，也未检测到环境变量 OPENAI_API_KEY")

    client = OpenAI(api_key=api_key, base_url=api_base)

    files: Iterable[Path] = sorted(input_dir.glob("*.jsonl"))
    if years:
        files = [p for p in files if _extract_year(p.name) in years]

    rich.print(f"🚀 待推理批次: {len(files)} 个 (input_dir={input_dir})")

    for fp in files:
        fp_out = output_dir / f"result_{fp.stem}.jsonl"
        if _is_output_complete(fp, fp_out):
            rich.print(f"[green]⏭️ {fp_out.name} 已完成，跳过[/green]")
            continue
        elif fp_out.exists():
            rich.print(f"[cyan]♻️ {fp_out.name} 不完整，继续续跑[/cyan]")

        _process_one_jsonl(fp, fp_out, client)

    rich.print(f"[bold green]🎉 全部批次推理完成，结果保存在 {output_dir}[/bold green]")

# ────────────────────────── CLI 入口 ────────────────────────── #
if __name__ == "__main__":
    import argparse, sys

    pa = argparse.ArgumentParser()
    pa.add_argument("--input_dir",  type=Path, required=True)
    pa.add_argument("--output_dir", type=Path, required=True)
    pa.add_argument("--years",      nargs="*", type=int, default=[])
    pa.add_argument("--api_key",    type=str, default=None)
    pa.add_argument("--api_base",   type=str, default=None)
    args = pa.parse_args()

    try:
        run_batches(
            args.input_dir,
            args.output_dir,
            years=args.years or None,
            api_key=args.api_key,
            api_base=args.api_base,
        )
    except Exception as exc:
        rich.print(f"[red]❌ 运行失败: {exc}[/red]")
        sys.exit(1)
