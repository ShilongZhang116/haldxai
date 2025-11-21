#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练 Aging-Classifier & 统一落盘
--------------------------------
CLI 示例
--------
python -m haldxai.workflow.train_aging_classifier             \
       --root  ~/Projects/HALDxAI-Project                      \
       --model-name aging_classifier_tfidf_lr_v1               \
       --neg-ratio 3                                           \
       --cv                # 输出五折 AUC

Notebook
--------
from haldxai.workflow.train_aging_classifier import run
run(project_root="~/Projects/HALDxAI-Project",
    model_name   ="aging_classifier_tfidf_lr_v1",
    neg_ratio    =3,
    show_cv      =True)
"""
from __future__ import annotations

from pathlib import Path
from typing   import List, Sequence

import yaml, typer
from dotenv import load_dotenv

from haldxai.init.config_utils            import load_config, update_config
from haldxai.modeling.aging_classifier.train import train_model     # 你已有的实现
from haldxai.modeling.common.save_utils      import save_model      # 你已有的实现


# ------------------------------------------------------------
# 内部：核心执行
# ------------------------------------------------------------
def _train_and_save(
    root         : Path,
    model_name   : str,
    pos_csv      : Path,
    neg_csv      : Path,
    aging_journals : Sequence[str],
    neg_ratio    : int   = 3,
    show_cv      : bool  = True,
    **kwargs,
):
    """真正的训练 + 落盘逻辑（CLI & Notebook 共用）"""

    # ---------- 1. 训练 ----------
    res = train_model(
        pos_csv       = pos_csv,
        neg_csv       = neg_csv,
        model_out     = Path("/dev/null"),    # 由 save_model 统一落盘
        aging_journals= list(aging_journals),
        neg_ratio     = neg_ratio,
        show_cv       = show_cv,
        **kwargs
    )
    model, aucs = res["model"], res["aucs"]

    # ---------- 2. 保存 ----------
    save_model(
        model        = model,
        model_name   = model_name,
        project_root = root,
        meta=dict(
            note         = "TF-IDF + LR，正负样本 1:{neg_ratio}".format(neg_ratio=neg_ratio),
            cv_mean_auc  = float(sum(aucs)/len(aucs)),
            tfidf_max_feat = kwargs.get("tfidf_max_feat", 5000),
            ngram          = kwargs.get("ngram", "1-2")
        )
    )

    # ---------- 3. 写回配置 ----------
    update_config(
        root / "configs" / "config.yaml",
        {f"models.{model_name}.saved": True}
    )

    print("🎉 训练 + 保存完成 -> models/{model_name}".format(model_name=model_name))
    return model


# ------------------------------------------------------------
# 公开给 Notebook 调用
# ------------------------------------------------------------
def run(
    project_root : str | Path,
    model_name   : str,
    neg_ratio    : int                 = 3,
    show_cv      : bool                = True,
    aging_journals: List[str] | None   = None,
):
    """
    Notebook 直接 `run(...)` 即可。
    其它高阶超参可通过 **kwargs 透传** 给 `train_model`。
    """
    root = Path(project_root).expanduser().resolve()

    cfg   = load_config(root / "configs" / "config.yaml")
    pos_csv = Path(cfg["articles_summary_aging-related_with_if_dir"]) / \
              "articles_summary_aging-related_with_if.csv"
    neg_csv = Path(cfg["articles_summary_not-aging-related_with_if_dir"]) / \
              "articles_summary_not-aging-related_with_if.csv"

    default_jc = ['The lancet. Healthy longevity', 'Nature aging', 'Aging cell',
                  'Ageing research reviews', 'Rejuvenation research',
                  'Aging', 'Age and ageing']
    aging_journals = aging_journals or default_jc

    return _train_and_save(root, model_name, pos_csv, neg_csv,
                           aging_journals, neg_ratio, show_cv)


# ------------------------------------------------------------
# CLI 封装 (Typer)
# ------------------------------------------------------------
cli = typer.Typer(help="训练 Aging-Classifier 并保存到 models/ 目录")

@cli.command()
def main(
    root       : Path = typer.Option(..., "--root", help="项目根目录"),
    model_name : str  = typer.Option(..., "--model-name"),
    neg_ratio  : int  = typer.Option(3,   "--neg-ratio", help="负样本 : 正样本"),
    show_cv    : bool = typer.Option(False, "--cv", help="是否打印 CV AUC"),
):
    """
    仅支持常用超参；更多高阶参数可在 Notebook 调 `run()` 时通过 **kwargs 传递。
    """
    load_dotenv(root / ".env", override=False)              # 如需额外 Key
    cfg = load_config(root / "configs" / "config.yaml")

    pos_csv = Path(cfg["articles_summary_aging-related_with_if_dir"]) / \
              "articles_summary_aging-related_with_if.csv"
    neg_csv = Path(cfg["articles_summary_not-aging-related_with_if_dir"]) / \
              "articles_summary_not-aging-related_with_if.csv"

    default_jc = ['The lancet. Healthy longevity', 'Nature aging', 'Aging cell',
                  'Ageing research reviews', 'Rejuvenation research',
                  'Aging', 'Age and ageing']

    _train_and_save(root, model_name, pos_csv, neg_csv,
                    default_jc, neg_ratio, show_cv)


if __name__ == "__main__":
    cli()        # `python -m haldxai.workflow.train_aging_classifier ...`
