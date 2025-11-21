#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
inspectors.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Quick-look utilities for HALDxAI databases

* preview_postgres() —— 列出所有表行数，并抽样展示前 N 行
* preview_neo4j()   —— 汇总节点/关系数量、标签/类型分布，并抽样示例

依赖:
  pip install psycopg2-binary neo4j python-dotenv pandas tabulate
"""

from __future__ import annotations

import os, textwrap, pprint, sys, itertools
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# ---------- 第三方 ----------
import psycopg2
from neo4j import GraphDatabase
import pandas as pd
from tabulate import tabulate

load_dotenv()    # 读取 .env

# ---------- Postgres 连接信息 ----------
PG_CONF = dict(
    host=os.getenv("PG_HOST", "localhost"),
    port=os.getenv("PG_PORT", "5432"),
    dbname=os.getenv("PG_DBNAME", "postgres"),
    user=os.getenv("PG_USER", "postgres"),
    password=os.getenv("PG_PASS", ""),
)

# ---------- Neo4j 连接信息 ----------
NEO4J_URI      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER     = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")


# ============================================================================
# 工具函数
# ============================================================================
def _print_df(df: pd.DataFrame, title: str = "") -> None:
    if title:
        print(f"\n{title}")
    print(tabulate(df, headers="keys", tablefmt="github", showindex=False))


def _pg_conn():
    return psycopg2.connect(**PG_CONF)


def _neo_driver():
    # Neo4j ≥5: encrypted="ENCRYPTION_OFF"
    return GraphDatabase.driver(
        NEO4J_URI,
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        encrypted="ENCRYPTION_OFF",
    )


# ============================================================================
# 1️⃣  Postgres 预览
# ============================================================================
def preview_postgres(schema: str = "hald", limit_per_table: int = 5) -> None:
    """
    打印 schema 中所有表行数，并对每张表抽样 `limit_per_table` 行。

    schema           —— 要查看的模式 (默认 hald)
    limit_per_table  —— 每个表 LIMIT N 行样本
    """
    try:
        with _pg_conn() as conn, conn.cursor() as cur:
            # 1. 列出所有 BASE TABLE
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = %s AND table_type = 'BASE TABLE'
                ORDER BY table_name;
                """,
                (schema,),
            )
            tables = [r[0] for r in cur.fetchall()]
            if not tables:
                print(f"⚠️  schema '{schema}' 中未找到表")
                return

            # 2. 行数统计
            counts = []
            for tbl in tables:
                cur.execute(f"SELECT COUNT(*) FROM {schema}.{tbl};")
                counts.append((tbl, cur.fetchone()[0]))
            df_counts = pd.DataFrame(counts, columns=["table", "rows"])
            _print_df(df_counts, f"📊  Postgres  – schema `{schema}`")

            # 3. 抽样
            for tbl in tables:
                cur.execute(f"SELECT * FROM {schema}.{tbl} LIMIT {limit_per_table};")
                rows = cur.fetchall()
                cols = [desc[0] for desc in cur.description]
                df_sample = pd.DataFrame(rows, columns=cols)
                _print_df(df_sample, f"🔹 Sample `{tbl}` ({limit_per_table} rows)")

    except Exception as e:
        print("❌  连接 Postgres 失败 –", e)


# ============================================================================
# 2️⃣  Neo4j 预览
# ============================================================================
def preview_neo4j(sample: int = 10) -> None:
    """
    汇总 Neo4j 节点/关系信息，并各抽样 `sample` 行。
    """
    try:
        with _neo_driver() as driver, driver.session(database="neo4j") as sess:
            # --- 总量 ---
            stats = sess.run(
                """
                CALL {
                  MATCH (n) RETURN count(n) AS nodes
                }
                CALL {
                  MATCH ()-[r]->() RETURN count(r) AS rels
                }
                RETURN nodes, rels
                """
            ).single()
            print(
                f"\n📌  Neo4j  nodes: {stats['nodes']:,}   relations: {stats['rels']:,}"
            )

            # --- 标签分布 ---
            df_labels = pd.DataFrame(
                sess.run(
                    """
                    MATCH (n) UNWIND labels(n) AS lab
                    RETURN lab AS label, count(*) AS cnt
                    ORDER BY cnt DESC
                    """
                ).data()
            )
            _print_df(df_labels, "🏷  Node labels")

            # --- 关系类型分布 ---
            df_reltypes = pd.DataFrame(
                sess.run(
                    """
                    MATCH ()-[r]->()
                    RETURN type(r) AS rel_type, count(*) AS cnt
                    ORDER BY cnt DESC
                    """
                ).data()
            )
            _print_df(df_reltypes, "🔗  Relation types")

            # --- 抽样节点 ---
            df_nodes = pd.DataFrame(
                sess.run(
                    """
                    MATCH (n)
                    RETURN
                      n.`node_id:ID` AS id,
                      labels(n)      AS labels,
                      n.name         AS name
                    LIMIT $N
                    """,
                    N=sample,
                ).data()
            )
            _print_df(df_nodes, f"🔹 Sample {sample} nodes")

            # --- 抽样关系 ---
            df_rels = pd.DataFrame(
                sess.run(
                    """
                    MATCH (a)-[r]->(b)
                    RETURN
                      r.relation_id        AS rid,
                      type(r)              AS type,
                      a.`node_id:ID`       AS src,
                      b.`node_id:ID`       AS tgt
                    LIMIT $N
                    """,
                    N=sample,
                ).data()
            )
            _print_df(df_rels, f"🔹 Sample {sample} relations")

    except Exception as e:
        print("❌  连接 Neo4j 失败 –", e)


# ============================================================================
# CLI 支持
# ============================================================================
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Quick inspector for Postgres & Neo4j",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--pg", action="store_true", help="preview Postgres")
    ap.add_argument("--neo", action="store_true", help="preview Neo4j")
    ap.add_argument("--schema", default="hald", help="Postgres schema")
    ap.add_argument("--limit", type=int, default=5, help="rows per PG table")
    ap.add_argument("--sample", type=int, default=10, help="sample rows in Neo4j")

    args = ap.parse_args()

    if args.pg:
        preview_postgres(schema=args.schema, limit_per_table=args.limit)
    if args.neo:
        preview_neo4j(sample=args.sample)
    if not args.pg and not args.neo:
        ap.print_help()
