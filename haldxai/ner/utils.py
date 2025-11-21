from pathlib import Path
import re
import os
import spacy
import subprocess
import sys
from spacy.cli import download as spacy_download
import nltk
from nltk.tokenize import sent_tokenize

# ---------- 路径 & 年份 ---------- #
def detect_years(path: Path, prefix: str) -> list[int]:
    years = set()
    for p in path.glob(f"{prefix}_Y*.csv"):
        m = re.search(r"_Y(\d{4})\.csv$", p.name)
        if m:
            years.add(int(m.group(1)))
    return sorted(years)

def build_save_path(base: Path, model: str, year: int, ext: str = "csv") -> Path:
    out_dir = base / model
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"ner_{year}.{ext}"

# ---------- SpaCy 模型 ---------- #
def ensure_spacy_model(model_name: str, local_repo: Path | str = "models/SciSpacy"):

    # ① 已安装？
    try:
        return spacy.load(model_name)
    except (OSError, IOError):
        pass                      # → 进入②

    # ② 本地 tar.gz？
    local_repo = Path(local_repo)
    if local_repo.exists():
        pattern = f"{model_name}*tar.gz"
        candidates = list(local_repo.glob(pattern))
        if candidates:
            pkg_path = candidates[0]
            print(f"📦 从本地安装 {pkg_path.name} …")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "pip", "install", pkg_path]
                )
                return spacy.load(model_name)
            except Exception as e:
                print(f"⚠️ 本地安装失败：{e}")

    # ③ 在线下载
    print(f"🌐 本地无模型 {model_name}，正在在线下载 …")
    spacy_download(model_name)
    return spacy.load(model_name)

# ---------- NLTK断句 ---------- #
def get_sentences_containing_offsets(text: str, offsets: list[tuple[int, int]]) -> str:
    """
    提取包含所有 offsets 的最小句子组合

    Args:
        text:     原始文本
        offsets:  [(start, end), ...]  — 实体在 text 中的索引区间

    Returns:
        str: 由一个或多个句子拼接而成，保证覆盖所有实体
    """
    # 确保 punkt 分词模型已就绪
    try:
        _ = nltk.data.find("tokenizers/punkt")
    except LookupError:  # 首次使用自动下载
        nltk.download("punkt", quiet=True)

    # 句子切分及其 span
    sentences = sent_tokenize(text)
    spans = list(nltk.tokenize.PunktSentenceTokenizer().span_tokenize(text))

    collected = []
    for s_start, s_end in spans:
        # 当前句子是否包含任何实体
        if any(s_start <= ent_start and ent_end <= s_end for ent_start, ent_end in offsets):
            collected.append(text[s_start:s_end].strip())

    if collected:
        return " ".join(collected)

    # fallback: 返回距离第一个实体最近的句子
    first_ent_start = offsets[0][0]
    nearest_span = min(spans, key=lambda sp: abs(first_ent_start - sp[0]))
    return text[nearest_span[0]:nearest_span[1]].strip()