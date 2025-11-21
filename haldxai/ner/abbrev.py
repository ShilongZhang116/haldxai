"""
abbrev.py
---------
✓ 解析实体中的括号缩写
✓ 无任何磁盘 I/O，方便多处复用
"""

from __future__ import annotations
import re
from typing import List, Dict


# ---------- 判断括号内容是否可能是英文缩写 ---------- #
def is_potential_abbreviation(text: str) -> bool:
    """
    极简启发式，可按需要自行加规则：
    1) 单字母且是 A-Z / a-z
    2) 2-8 个字符，至少 50 % 为字母，且不是纯数字
    """
    if not text:
        return False
    text = text.strip()

    if len(text) == 1 and text.isalpha():
        return True

    if 2 <= len(text) <= 8:
        alpha = sum(c.isalpha() for c in text)
        digit = sum(c.isdigit() for c in text)
        if alpha == 0 or digit == len(text):
            return False
        return alpha / len(text) >= 0.5
    return False


# ---------- 主解析函数 ---------- #
def parse_entity_abbreviation(entity: str) -> Dict:
    """
    返回字典示例：
    {
        "original_text": "reactive oxygen species (ROS)",
        "main_text":     "reactive oxygen species",
        "details":       [ {"content":"ROS", "type":"abbreviation"} ]
    }
    """
    result = {
        "original_text": entity,
        "main_text": entity,
        "details": []        # List[{"content": str, "type": "abbreviation"|"info"}]
    }

    if "(" not in entity:
        return result

    # 尝试“末尾单括号”模式
    m = re.match(r"^(.*?)\s*\(([^)]+)\)\s*$", entity)
    if m:
        head, inside = m.group(1).strip(), m.group(2).strip()
        result["main_text"] = head or entity
        result["details"].append({
            "content": inside,
            "type": "abbreviation" if is_potential_abbreviation(inside) else "info"
        })
        return result

    # 复杂情况：逐 token 处理
    tokens = re.split(r"(\(.*?\))", entity)
    main_builder: List[str] = []

    for tok in tokens:
        if tok.startswith("(") and tok.endswith(")"):
            core = tok[1:-1].strip()
            result["details"].append({
                "content": core,
                "type": "abbreviation" if is_potential_abbreviation(core) else "info"
            })
        elif tok.strip():
            main_builder.append(tok)

    if main_builder:
        result["main_text"] = "".join(main_builder).strip()
    return result
