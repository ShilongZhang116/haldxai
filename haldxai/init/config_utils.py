#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高频使用的 **配置 / 环境文件** 工具

功能一览
---------
1. `init_config`      —— 当 `config.yaml` 不在时写入默认配置
2. `update_config`    —— 在已有 `config.yaml` 中补齐缺失字段
3. `load_config`      —— 读取 YAML → `dict`
4. `save_config`      —— 把 `dict` 写回 YAML
5. `set_config`       —— 单字段写入（支持点号层级）
6. `write_env`        —— 生成 `.env`（可选覆盖）
7. `init_project`     —— 一键写 `config.yaml` + `.env`
8. `show_config`      —— 在 Notebook 里友好展示配置
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Dict

import os
import textwrap
import yaml
from dotenv import dotenv_values

# ------------------------------------------------------------
# 🔧 帮助函数：获取项目根
# ------------------------------------------------------------
def _resolve_root(root: str | Path | None) -> Path:
    """
    若显式传入 root，则使用；否则退回到“当前文件向上两级”。
    """
    if root is None:
        return Path(__file__).resolve().parents[2]
    return Path(root).expanduser().resolve()

# ------------------------------------------------------------
# 读 / 写 / 单字段更新
# ------------------------------------------------------------
def load_config(cfg_path: str | Path, *, project_root: str | Path | None = None) -> dict:
    cfg_path = _resolve_root(project_root) / cfg_path
    if not cfg_path.exists():
        return {}
    return yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}


def save_config(
    cfg: Mapping[str, Any],
    cfg_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> None:
    cfg_path = _resolve_root(project_root) / cfg_path
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        yaml.dump(dict(cfg), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    print(f"💾  config.yaml 已保存 → {cfg_path}")


def set_config(
    key: str,
    value: Any,
    cfg_path: str | Path = "configs/config.yaml",
    *,
    project_root: str | Path | None = None,
    create: bool = True,
) -> None:
    cfg = load_config(cfg_path, project_root=project_root)

    # 支持点号层级
    if "." in key:
        levels = key.split(".")
        cur = cfg
        for lv in levels[:-1]:
            if lv not in cur:
                if not create:
                    raise KeyError(f"键 {lv} 不存在且 create=False")
                cur[lv] = {}
            cur = cur[lv]
        cur[levels[-1]] = value
    else:
        cfg[key] = value

    save_config(cfg, cfg_path, project_root=project_root)


# ------------------------------------------------------------
# 写入 / 更新配置（保持旧接口，但加 root 参数）
# ------------------------------------------------------------
def init_config(
    config_path: str | Path,
    default_cfg: Dict[str, Any],
    *,
    project_root: str | Path | None = None,
) -> None:
    config_path = _resolve_root(project_root) / config_path
    config_path.parent.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text(
            yaml.dump(default_cfg, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        print(f"✅  已创建新的 config.yaml → {config_path}")
    else:
        print(f"🟡  {config_path} 已存在，跳过写入")


def update_config(
    config_path: str | Path,
    new_cfg: Dict[str, Any],
    *,
    project_root: str | Path | None = None,
) -> None:
    cfg = load_config(config_path, project_root=project_root)
    updated = False
    for k, v in new_cfg.items():
        if k not in cfg:
            cfg[k] = v
            updated = True
    if updated:
        save_config(cfg, config_path, project_root=project_root)
        print("🛠  缺失字段已补齐")
    else:
        print("🔍  当前配置已完整")


# ------------------------------------------------------------
# 生成 .env
# ------------------------------------------------------------
_DEFAULT_ENV = textwrap.dedent(
    """\
    # -------- API KEYS --------
    PUBMED_API_KEY=
    DEEPSEEK_API_KEY=
    BIOPORTAL_API_KEY=

    # -------- EMAIL --------
    PUBMED_EMAIL=

    # -------- 可选路径重写 --------
    # PROJECT_ROOT=/abs/path/to/HALDxAI
"""
)


def write_env(
    env_path: str | Path = ".env",
    template: str = _DEFAULT_ENV,
    *,
    project_root: str | Path | None = None,
    force: bool = False,
) -> None:
    env_path = _resolve_root(project_root) / env_path
    if env_path.exists() and not force:
        print(f"🟡  {env_path} 已存在（--force 覆盖）")
        return
    env_path.write_text(template, encoding="utf-8")
    print(f"✅  已写入 .env → {env_path}")


# ------------------------------------------------------------
# 一键初始化
# ------------------------------------------------------------
_DEFAULT_PROJECT_CFG: Dict[str, Any] = {
    "project_root": "",  # 初始化时再写入绝对路径
    "data_dir": "data",
    "raw_data_dir": "data/raw",
    "intermediate_dir": "data/interi",
    "model_dir": "models",
    "log_dir": "logs",
    "config_dir": "configs",
    "api": {
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "timeout": 30,
        },
        "bioportal": {
            "base_url": "https://data.bioontology.org/search",
            "page_size": 10,
        },
    },
    "batch": {"max_workers": 16, "chunk_size": 200},
}


def init_project(
    project_root: str | Path,
    *,
    force: bool = False,
    default_cfg: Dict[str, Any] | None = None,
    env_template: str = _DEFAULT_ENV,
) -> None:
    """
    一键生成 / 覆盖 `config.yaml` 与 `.env`

    Parameters
    ----------
    project_root : 项目根目录（必须显式给出，Notebook 环境最安全）
    force        : True 则覆盖已存在的同名文件
    """
    root = _resolve_root(project_root)
    cfg_path = root / "configs" / "config.yaml"
    env_path = root / ".env"

    # 填入真实的 project_root 后再写
    cfg_to_write = (default_cfg or _DEFAULT_PROJECT_CFG).copy()
    cfg_to_write["project_root"] = str(root)

    if not cfg_path.exists() or force:
        init_config(cfg_path, cfg_to_write, project_root=root)
    else:
        print("🟡  config.yaml 已存在（用 force=True 可覆盖）")

    write_env(env_path, env_template, project_root=root, force=force)
    print("🎉  项目初始化完成 →", root)


# ------------------------------------------------------------
# Notebook 友好展示
# ------------------------------------------------------------
def show_config(
    project_root: str | Path,
    *,
    show_env: bool = True,
) -> None:
    """在 Notebook / 终端友好打印 `config.yaml` & `.env`"""
    import pprint

    root = _resolve_root(project_root)
    cfg_path = root / "configs" / "config.yaml"
    env_path = root / ".env"

    print(f"📄  {cfg_path}")
    pprint.pprint(load_config(cfg_path), width=88, compact=False)

    if show_env and env_path.exists():
        print(f"\n🔑  {env_path}")
        env = {k: v for k, v in dotenv_values(env_path).items() if v}
        pprint.pprint(env, width=88, compact=False)


# ------------------------------------------------------------
# CLI 入口
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    pa = argparse.ArgumentParser(description="初始化 HALDxAI 项目（生成 config.yaml & .env）")
    pa.add_argument("project_root", help="目标项目根目录（必填）")
    pa.add_argument("--force", action="store_true", help="覆盖已存在文件")
    args = pa.parse_args()

    init_project(args.project_root, force=args.force)
