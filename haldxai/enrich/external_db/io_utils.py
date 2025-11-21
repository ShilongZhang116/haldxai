# haldxai/enrich/external_db/io_utils.py
import pandas as pd, chardet
import csv
from pathlib import Path


def read_tsv_robust(path, max_bad=1000):
    """
    自动识别文件分隔符（逗号或tab）和编码，尝试使用 pandas 正确读取。
    遇到结构混乱时回退到手动 split。
    """
    path = Path(path)
    # ---- 检测编码 ----
    with open(path, 'rb') as f:
        enc_guess = chardet.detect(f.read(20000))['encoding'] or 'utf-8'

    # ---- 预览前两行内容，判断分隔符 ----
    with open(path, 'r', encoding=enc_guess, errors='ignore') as f:
        lines = [f.readline() for _ in range(2)]
    sep = ',' if lines[0].count(',') > lines[0].count('\t') else '\t'

    try:
        return pd.read_csv(
            path, sep=sep, encoding=enc_guess,
            quoting=csv.QUOTE_MINIMAL,  # 支持引号包裹字段
            on_bad_lines="skip",  # pandas ≥ 1.3
            engine="python"
        )
    except UnicodeDecodeError as e:
        return pd.read_csv(
            path, sep=sep, encoding='gbk',
            quoting=csv.QUOTE_MINIMAL,  # 支持引号包裹字段
            on_bad_lines="skip",  # pandas ≥ 1.3
            engine="python"
        )

    except Exception as e:
        # 回退：手动解析
        print(f"[WARN] fallback parsing: {path.name}, error: {e}")
        good_rows, bad = [], 0
        header = lines[0].rstrip("\n").split(sep)
        with open(path, 'r', encoding=enc_guess, errors='ignore') as f:
            next(f)  # skip header
            for line in f:
                parts = line.rstrip("\n").split(sep)
                if len(parts) == len(header):
                    good_rows.append(parts)
                else:
                    bad += 1
                    if bad >= max_bad:
                        break
        df = pd.DataFrame(good_rows, columns=header)
        print(f"[INFO] {path.name}: skipped {bad} bad lines")

        return df