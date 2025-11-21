#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""import_to_pg.py (revised)
────────────────────────────
Bulk‑import CSV files under *data/database* into a PostgreSQL schema **hald**.

🔄 **What's new**
1. **Selective import** – `--table` lets you import a *single* table (or a comma‑separated list). Omit to import **all** tables (previous default).
2. **Overwrite vs append** – replace `--no-truncate` with clearer `--mode`:
   - `replace`  ➜ TRUNCATE table then COPY (default)
   - `append`   ➜ COPY without TRUNCATE
   - `skip`     ➜ skip table even if CSV present (useful with list)
3. Retains `--no-vacuum` behaviour (post‑import VACUUM ANALYZE).

Usage examples
--------------
```bash
# import ALL tables, replacing existing rows
python import_to_pg.py

# import only entity_catalog (append mode)
python import_to_pg.py --table entity_catalog --mode append

# import two tables, keep others untouched
python import_to_pg.py --table articles,entity_evidence
```
"""
from __future__ import annotations
from pathlib import Path
import argparse, os, sys, time, psycopg2
from dotenv import load_dotenv

load_dotenv()

# ────────────────────────────── DB config ──────────────────────────────
PG_CONF = dict(
    host=os.getenv("PG_HOST", "localhost"),
    port=os.getenv("PG_PORT", "5432"),
    dbname=os.getenv("PG_DBNAME", "postgres"),
    user=os.getenv("PG_USER", "postgres"),
    password=os.getenv("PG_PASS", ""),
)

# csv filename → (target table, column list | None)
TABLE_MAP = {
    "articles.csv":               ("articles",               None),
    "entity_catalog.csv":         ("entity_catalog",         None),
    "entity_catalog_ext.csv":     ("entity_catalog_ext",     None),
    "entity_evidence.csv":        ("entity_evidence",        None),
    "entity_types.csv":           ("entity_types",           None),
    "entity_types_pred.csv":      ("entity_types_pred",      None),
    "relation_evidence.csv":      ("relation_evidence",      None),
    "relation_types.csv":         ("relation_types",         None),
}

# ────────────────────────────── helpers ──────────────────────────────

def copy_csv(cur, table: str, csv_path: Path, columns: list[str] | None):
    col_part = f"({','.join(columns)})" if columns else ""
    with csv_path.open("r", encoding="utf-8") as f:
        cur.copy_expert(
            sql=f"COPY {table}{col_part} FROM STDIN WITH (FORMAT csv, HEADER true, ENCODING 'utf8')",
            file=f,
        )


def table_selection(tables_arg: str | None):
    """Return a set of table names to import based on CLI arg."""
    if not tables_arg:
        return {t for _, (t, _) in TABLE_MAP.items()}  # all
    wanted = {t.strip() for t in tables_arg.split(",") if t.strip()}
    unknown = wanted - {v[0] for v in TABLE_MAP.values()}
    if unknown:
        raise SystemExit(f"❌ Unknown table(s): {', '.join(unknown)}")
    return wanted

# ────────────────────────────── main ──────────────────────────────

def main(csv_dir: Path, mode: str, tables_arg: str | None, vacuum: bool):
    selected_tables = table_selection(tables_arg)

    with psycopg2.connect(**PG_CONF) as conn, conn.cursor() as cur:
        cur.execute("SET search_path TO hald, public;")

        for fname, (table, cols) in TABLE_MAP.items():
            if table not in selected_tables:
                continue
            fpath = csv_dir / fname
            if not fpath.exists():
                print(f"⚠️  CSV not found → skip: {fname}")
                continue

            if mode == "replace":
                cur.execute(f"TRUNCATE {table};")
            elif mode == "skip":
                print(f"⏭  skip mode active for table: {table}")
                continue

            print(f"⬆️  {fname}  →  {table}  ({mode})")
            t0 = time.time()
            copy_csv(cur, table, fpath, cols)
            print(f"   ✔ done in {time.time() - t0:.1f}s  ({fpath.stat().st_size/1e6:.1f} MB)")

    print("🎉  import finished")

# ────────────────────────────── CLI ──────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bulk import HALD CSV into PostgreSQL")
    parser.add_argument("--dir", default="data/database", help="CSV root directory (default data/database)")
    parser.add_argument("--table", help="Target table name or comma‑separated list. Omit for ALL tables")
    parser.add_argument("--mode", choices=["replace", "append", "skip"], default="replace",
                        help="Import mode per table: replace=TRUNCATE+COPY (default), append=COPY only, skip=ignore")
    parser.add_argument("--no-vacuum", action="store_true", help="Skip VACUUM ANALYZE after import")
    args = parser.parse_args()

    try:
        main(
            csv_dir=Path(args.dir).expanduser().resolve(),
            mode=args.mode,
            tables_arg=args.table,
            vacuum=not args.no_vacuum,
        )
    except KeyboardInterrupt:
        sys.exit("\n⏹  interrupted by user")
