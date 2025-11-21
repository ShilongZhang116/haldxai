# -*- coding: utf-8 -*-
"""
PostgreSQL helpers: wipe & bulk-import   (with progress bar)
"""
from __future__ import annotations
import csv, os, re
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable

import pandas as pd
import psycopg2
from psycopg2 import sql
from dotenv import load_dotenv
from tqdm import tqdm          # ← 新增

# ────────── 环境变量 ──────────
load_dotenv()
PG_CONF: Dict[str, str | int] = {
    "host":     os.getenv("PG_HOST", "localhost"),
    "port":     int(os.getenv("PG_PORT", 5432)),
    "dbname":   os.getenv("PG_DBNAME", "postgres"),
    "user":     os.getenv("PG_USER", "postgres"),
    "password": os.getenv("PG_PASS", "")
}

# ────────── DDL / COPY 工具 ──────────
def _guess_pg_type(col: pd.Series) -> str:
    col = col.dropna()
    if pd.api.types.is_integer_dtype(col):
        return "BIGINT"
    if pd.api.types.is_float_dtype(col):
        return "DOUBLE PRECISION"
    if pd.api.types.is_bool_dtype(col):
        return "BOOLEAN"
    return "TEXT"

def _copy_df(cur, table: str, df: pd.DataFrame) -> None:
    """高速 COPY（DataFrame → StringIO → COPY ... FROM STDIN）"""
    buf = StringIO()
    df.to_csv(buf, index=False, header=False,
              sep="\t", na_rep="\\N", quoting=csv.QUOTE_MINIMAL)
    buf.seek(0)
    cols = ','.join(f'"{c}"' for c in df.columns)
    cur.copy_expert(
        sql.SQL(
            "COPY {} ({}) FROM STDIN WITH (FORMAT CSV, DELIMITER '\t', NULL '\\N')"
        ).format(sql.Identifier(table), sql.SQL(cols)),
        buf
    )

# ────────── 清空 ──────────
def wipe_schema(schema: str = "public"):
    with psycopg2.connect(**PG_CONF) as conn:
        conn.autocommit = True
        cur = conn.cursor()
        cur.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE;')
        cur.execute(f'CREATE SCHEMA "{schema}";')
        print(f"✅ [{schema}] schema dropped & recreated")

def wipe(drop_schema: bool = False) -> None:
    """
    清空 public schema
    - drop_schema=True 时直接 `DROP SCHEMA public CASCADE; CREATE SCHEMA public`
      （最快且最干净）
    - 否则逐个 DROP table / view / seq
    """
    with psycopg2.connect(**PG_CONF) as conn:
        conn.autocommit = True
        cur = conn.cursor()

        if drop_schema:
            cur.execute("DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;")
            print("✅ [PG] public schema dropped & recreated")
            return

        # 普通表 / 视图
        cur.execute("""
            SELECT table_name, table_type
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        for name, typ in cur.fetchall():
            cur.execute(sql.SQL("DROP {} IF EXISTS {} CASCADE")
                        .format(sql.SQL(typ.replace(" ", "_")), sql.Identifier(name)))

        # 物化视图
        cur.execute("SELECT matviewname FROM pg_matviews WHERE schemaname='public'")
        for (mv,) in cur.fetchall():
            cur.execute(sql.SQL("DROP MATERIALIZED VIEW IF EXISTS {} CASCADE")
                        .format(sql.Identifier(mv)))

        # 序列
        cur.execute("""
            SELECT sequence_name FROM information_schema.sequences
            WHERE sequence_schema='public'
        """)
        for (seq,) in cur.fetchall():
            cur.execute(sql.SQL("DROP SEQUENCE IF EXISTS {} CASCADE")
                        .format(sql.Identifier(seq)))

        print("✅ [PG] public schema cleaned (tables/views/sequences removed)")
