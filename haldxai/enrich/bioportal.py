"""
bioportal.py (v2)
-----------------
* 读取  ner_dictionary/parsed_entities_*.csv
* 并发查询 BioPortal 并补全 canonical / synonyms / 定义
* 带缓存 + checkpoint，可断点续跑
* 既可 CLI，也可 Notebook 调用

用 CLI：
    python -m haldxai.enrich.bioportal \
        --tasks AgingRelated-DeepSeekV3 en_ner_bc5cdr_md \
        --api_key xxx                          # 不传就看环境变量 BIOPORTAL_API_KEY
        --out_dir  /tmp/mydict                 # 可选
"""

from __future__ import annotations
from pathlib import Path
import os, re, json, math, time, yaml, urllib.parse, requests
from typing import List, Dict, Any

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception_type
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────
# 0. 默认路径 & 配置
# ──────────────────────────────────────────────────────────────
_PRJ = Path(__file__).resolve().parents[2]          # …/HALDxAI-Project
_CFG = yaml.safe_load((_PRJ / "configs/config.yaml").read_text(encoding="utf-8"))

# 默认目录
_DEFAULT_DICT_DIR = _PRJ / "data/ner_dict"
_DEFAULT_CACHE_DIR = _DEFAULT_DICT_DIR / "bioPortal"
_DEFAULT_CACHE_DIR.mkdir(exist_ok=True, parents=True)

# 正则
_LEADING_CHARS = re.compile(r'^[^\w\u4e00-\u9fff]+')
_ONLY_NUM_SYM  = re.compile(r'[\d\W_]+$')

load_dotenv(_PRJ / ".env", override=False)

# BioPortal
BIO_API  = "https://data.bioontology.org/search"
API_KEY  = os.getenv("BIOPORTAL_API_KEY", "")       # 也可 CLI 参数

# 并发参数（可在 run_bioportal() 覆写）
MAX_WORKERS   = 16
CHUNK_SIZE    = 200
SLEEP_BETWEEN = 3   # 秒


# ──────────────────────────────────────────────────────────────
# 1. BioPortal 查询
# ──────────────────────────────────────────────────────────────
@retry(stop=stop_after_attempt(3),
       wait=wait_fixed(2),
       retry=retry_if_exception_type(requests.exceptions.RequestException))
def _search_bioportal(q: str, api_key: str, page_size: int = 5) -> dict | None:
    params = {
        "q": q, "require_exact_match": "false",
        "pagesize": page_size, "suggest": "true"
    }
    hdr = {"Authorization": f"apikey token={api_key}"}
    r = requests.get(f"{BIO_API}?{urllib.parse.urlencode(params)}",
                     headers=hdr, timeout=10)
    if r.status_code == 200:
        return r.json()
    if r.status_code == 429:          # 限流
        time.sleep(5)
    r.raise_for_status()
    return None


def _parse_result(query: str, raw: dict | None) -> dict:
    if not raw or "collection" not in raw:
        return {"query": query, "count": 0, "items": []}
    items = []
    for itm in raw["collection"]:
        items.append({
            "pref_label":  itm.get("prefLabel", ""),
            "synonyms":    itm.get("synonym", []),
            "definitions": itm.get("definition", []),
            "ontology":    itm.get("links", {}).get("ontology"),
            "class_iri":   itm.get("@id"),
            "score":       itm.get("score", 0),
        })
    return {"query": query, "count": len(items), "items": items}


# ──────────────────────────────────────────────────────────────
# 2. 断点续跑用的轻量缓存
# ──────────────────────────────────────────────────────────────
def _load_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}

def _dump_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


# ──────────────────────────────────────────────────────────────
# 3. 核心：批量处理一个实体列表
# ──────────────────────────────────────────────────────────────
def _clean_text(s: str) -> str | None:
    if not isinstance(s, str):
        return None
    s = (s.replace(",", ";").replace('"', "")
          .replace("'", "").replace("\n", " ").strip())
    s = _LEADING_CHARS.sub("", s).lstrip()
    if not s or _ONLY_NUM_SYM.fullmatch(s):
        return None
    return s

def _process_entity(text: str,
                    cache_rev: dict, cache_can: dict,
                    cache_search: dict,
                    api_key: str) -> tuple[str, dict]:
    norm = text.lower().strip()
    if norm in cache_rev:                 # 命中缓存
        canon = cache_rev[norm]
        return text, {
            "canonical": canon,
            "all_synonyms": list(cache_can[canon]),
            "search_result": cache_search.get(canon, {})
        }

    raw = _search_bioportal(text, api_key)
    parsed = _parse_result(text, raw)

    if parsed["count"]:                   # 取首条
        itm = parsed["items"][0]
        canon = itm["pref_label"] or text
        syns  = set(itm["synonyms"]) | {canon}
    else:
        canon = text
        syns  = {text}

    # 写三张缓存
    for s in syns:
        cache_rev[s.lower()] = canon
    cache_can[canon] = syns
    cache_search[canon] = parsed

    return text, {
        "canonical": canon,
        "all_synonyms": list(syns),
        "search_result": parsed
    }


# ──────────────────────────────────────────────────────────────
# 4. 对外主函数
# ──────────────────────────────────────────────────────────────
def run_bioportal(
    tasks: List[str],
    api_key: str | None = None,
    dict_dir: Path | str | None = None,
    cache_dir: Path | str | None = None,
    max_workers: int = MAX_WORKERS,
    chunk_size: int = CHUNK_SIZE,
    sleep_between: int = SLEEP_BETWEEN,
) -> Path:
    """
    tasks      : ['AgingRelated-DeepSeekV3', 'en_ner_bc5cdr_md', ...]
    api_key    : 不传 → 环境变量 BIOPORTAL_API_KEY
    dict_dir   : parsed_entities_*.csv 所在目录；默认 data/ner_dictionary
    cache_dir  : 保存 cache / checkpoint / final.json 的目录
    返回值     : final_entity_results.json 的路径
    """
    api_key = api_key or os.getenv("BIOPORTAL_API_KEY")
    if not api_key:
        raise RuntimeError("请提供 BioPortal API_KEY，或设环境变量 BIOPORTAL_API_KEY")

    dict_dir  = Path(dict_dir  or _DEFAULT_DICT_DIR)
    cache_dir = Path(cache_dir or _DEFAULT_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)

    f_cache       = cache_dir / "bioportal_cache.json"
    f_ckpt        = cache_dir / "checkpoint.json"
    f_final       = cache_dir / "final_entity_results.json"

    # --- 恢复缓存 ---
    cache_rev   = _load_json(f_cache).get("reverse_synonym_map", {})
    cache_can   = {k: set(v) for k, v in _load_json(f_cache).get("canonical_map", {}).items()}
    cache_search= _load_json(f_cache).get("search_results_map", {})
    entity_dict = _load_json(f_ckpt)      # 可能为空

    # --- 读取 parsed_entities_*.csv ---
    dfs = []
    for t in tasks:
        fp = dict_dir / f"parsed_entities_{t}.csv"
        if not fp.exists():
            print(f"⚠️ 未找到 {fp.name}，跳过")
            continue
        dfs.append(pd.read_csv(fp))
    if not dfs:
        raise RuntimeError("❌ 未读取到任何 parsed_entities_*.csv")

    df_all = pd.concat(dfs, ignore_index=True)
    df_all["main_text"] = df_all["main_text"].apply(_clean_text)
    todo = df_all["main_text"].dropna().unique()
    todo = [t for t in todo if t not in entity_dict]
    print(f"📝 待处理实体数: {len(todo)}")

    # --- 分批并发 ---
    for idx in range(0, len(todo), chunk_size):
        chunk = todo[idx: idx+chunk_size]
        print(f"\n[Chunk {idx//chunk_size+1}] 处理 {len(chunk)} 条")

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = [
                ex.submit(_process_entity, ent,
                          cache_rev, cache_can, cache_search, api_key)
                for ent in chunk
            ]
            for f in tqdm(as_completed(futs), total=len(futs), desc="BioPortal"):
                ent, info = f.result()
                entity_dict[ent] = info

        # 写 checkpoint & cache
        _dump_json(f_ckpt, entity_dict)
        _dump_json(f_cache, {
            "reverse_synonym_map": cache_rev,
            "canonical_map": {k: list(v) for k, v in cache_can.items()},
            "search_results_map": cache_search
        })
        if idx + chunk_size < len(todo):
            print(f"休眠 {sleep_between}s...")
            time.sleep(sleep_between)

    # --- 最终输出 ---
    _dump_json(f_final, entity_dict)
    print(f"\n✅ 全部完成！结果 → {f_final}")
    return f_final


# ──────────────────────────────────────────────────────────────
# 5. CLI
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse, sys, textwrap
    pa = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            批量查询 BioPortal 并生成实体同义词字典
            示例:
              python -m haldxai.enrich.bioportal \\
                    --tasks AgingRelated-DeepSeekV3 en_ner_bc5cdr_md \\
                    --api_key XXX
        """))
    pa.add_argument("--tasks", nargs="+", required=True, help="parsed_entities_<task>.csv 的 <task> 列表")
    pa.add_argument("--api_key")
    pa.add_argument("--dict_dir")
    pa.add_argument("--cache_dir")
    pa.add_argument("--max_workers", type=int, default=MAX_WORKERS)
    pa.add_argument("--chunk_size",  type=int, default=CHUNK_SIZE)
    pa.add_argument("--sleep",       type=int, default=SLEEP_BETWEEN)
    args = pa.parse_args()

    try:
        run_bioportal(
            tasks=args.tasks,
            api_key=args.api_key,
            dict_dir=args.dict_dir,
            cache_dir=args.cache_dir,
            max_workers=args.max_workers,
            chunk_size=args.chunk_size,
            sleep_between=args.sleep,
        )
    except Exception as e:
        print(f"❌ 失败: {e}")
        sys.exit(1)
