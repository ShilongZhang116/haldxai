# haldxai/utils/id_utils.py
from __future__ import annotations
import hashlib
from typing import Dict, Optional

def alloc_id(name2id: Dict[str, str], name: str | None) -> Optional[str]:
    """
    根据实体名称分配（或返回已有） entity_id。
    如果 name 为空，则返回 None。

    Parameters
    ----------
    name2id : dict[str, str]
        已有的 name→id 映射（会原地更新）
    name : str | None
        实体原始名称

    Returns
    -------
    str | None
        entity_id，例如 "Entity-4f7a1c3b9e"，或 None
    """
    # ① 判空
    if name is None:
        return None
    name = str(name).strip()
    if not name:
        return None

    # ② 标准化 key
    key = name.lower()

    # ③ 若已存在直接返回
    if key in name2id:
        return name2id[key]

    # ④ 新生成
    new_id = "Entity-" + hashlib.md5(key.encode("utf-8")).hexdigest()[:10]
    name2id[key] = new_id
    return new_id
