"""
haldxai.enrich.external_db.cli
==============================

统一入口：一键或按需执行 `build_xxx_std.py` 里的 `build_xxx()` 主函数。
用法示例::

    # 查看帮助
    python -m haldxai.enrich.external_db.cli --help

    # 仅跑 mesh + hagr
    python -m haldxai.enrich.external_db.cli mesh hagr

    # 一键全部跑（自动跳过已存在的 _std 文件）
    python -m haldxai.enrich.external_db.cli all --root F:/Project/HALDxAI --force
"""
from __future__ import annotations
import importlib, inspect, pkgutil, sys
from pathlib import Path
from typing import List, Dict

import typer

app = typer.Typer(add_completion=False, help="Build *_std.csv for external DBs")

# ─────────────────────────────────────────────────────────────
# 1. 自动发现所有 build_xxx_std.py 并收集 build_xxx() 函数
# ─────────────────────────────────────────────────────────────
_THIS_PKG = __name__.rsplit(".", 1)[0]            # haldxai.enrich.external_db
_FUNCS: Dict[str, callable] = {}

for modinfo in pkgutil.iter_modules(sys.modules[_THIS_PKG].__path__):
    name = modinfo.name
    if name.startswith("build_") and name.endswith("_std"):
        mod = importlib.import_module(f"{_THIS_PKG}.{name}")
        # 约定每个脚本里只有 1 个以 build_ 开头的公开函数
        for attr_name, attr in inspect.getmembers(mod, inspect.isfunction):
            if attr_name.startswith("build_") and attr.__module__ == mod.__name__:
                key = attr_name.replace("build_", "").replace("_std", "")  # 例如 hagr
                _FUNCS[key] = attr
                break

# ─────────────────────────────────────────────────────────────
def _run_one(tag: str, project_root: Path, force: bool):
    func = _FUNCS.get(tag)
    if func is None:
        typer.echo(f"❌ 未找到构建器: {tag}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"🚀 [{tag}] running …")
    func(project_root, force=force)   # 各 build_xxx(project_root, force)
    typer.echo(f"✅ [{tag}] done.\n")

# ─────────────────────────────────────────────────────────────
@app.command(help="按名称执行一个或多个构建器；可用 special name 'all'")
def main(
    names: List[str] = typer.Argument(
        ...,
        help="要执行的构建器名称（比如 mesh hagr）或 'all'"
    ),
    root: Path = typer.Option(
        ..., "--root", "-r", exists=True, file_okay=False,
        help="项目根目录（即包含 data/bio_corpus 的目录）"
    ),
    force: bool = typer.Option(
        False, "--force", "-f",
        help="如为 True 则无视已存在的 _std 文件，强制重跑"
    )
):
    if "all" in names:
        targets = sorted(_FUNCS)
    else:
        targets = names
    typer.echo(f"🧮 计划执行: {', '.join(targets)}")
    for t in targets:
        _run_one(t, project_root=root, force=force)

# 允许  `python -m haldxai.enrich.external_db.cli …`  直接调用
if __name__ == "__main__":
    app()
