# ─────────────────────────────────────────────────────────
#  nodes.py   —— 生成 data/database/nodes.csv
# ─────────────────────────────────────────────────────────
from __future__ import annotations

import json
import re
import os
from pathlib import Path
from typing import Dict

import pandas as pd

from dotenv import load_dotenv
from neo4j import GraphDatabase

# ════════════════════════════════════════════════════════
# 映射、工具
# ════════════════════════════════════════════════════════
_MODEL_LABEL_MAP: Dict[str, str] = {
    # DeepSeek
    "AgingRelated-DeepSeekR1-32B": "DeepSeekR1_32B",
    "AgingRelated-DeepSeekR1-7B":  "DeepSeekR1_7B",
    "AgingRelated-DeepSeekV3":     "DeepSeekV3",
    "JCRQ1-IF10-DeepSeekR1-32B":   "DeepSeekR1_32B",
    "JCRQ1-IF10-DeepSeekR1-7B":    "DeepSeekR1_7B",
    "JCRQ1-IF10-DeepSeekV3":       "DeepSeekV3",
    # SciSpacy
    "en_ner_bc5cdr_md":     "SciSpacy_BC5CDR",
    "en_ner_bionlp13cg_md": "SciSpacy_BioNLP13CG",
    "en_ner_jnlpba_md":     "SciSpacy_JNLPBA",
    None: "UnknownModel",
}


def _clean_src_file(fname: str) -> str:
    """hald_ageanno__xxx__std.csv → Ageanno"""
    base = re.sub(r"^hald_|__std\.csv$", "", fname)
    return base.split("__")[0].capitalize()


def _merge_labels(series: pd.Series) -> str:
    """去重后用 | 连接；始终带 'Entity'"""
    return "|".join(sorted({"Entity", *series.dropna()}))


load_dotenv()
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

def _driver():
    return GraphDatabase.driver(NEO4J_URI,
                                auth=(NEO4J_USER, NEO4J_PASSWORD),
                                encrypted=False)

# ────────── wipe ──────────
def wipe() -> None:
    with _driver().session() as sess:
        sess.run("MATCH (n) DETACH DELETE n")
    print("✅ [Neo4j] all nodes & relations deleted")
