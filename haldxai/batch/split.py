from pathlib import Path
import pandas as pd
from .filter import get_filter

def split_year_to_batches(
    year: int,
    filter_method: str,
    task_name: str,
    input_dir: Path,
    prefix: str,
    output_dir: Path,
    batch_size: int = 200,
) -> None:
    csv_file = input_dir / f"{prefix}_Y{year}.csv"
    if not csv_file.exists():
        print(f"⚠️ 找不到 {csv_file.name}")
        return

    df = pd.read_csv(csv_file)
    df = get_filter(filter_method)(df)          # ← 调用过滤器

    if df.empty:
        print(f"⚠️ {year} 无符合条件数据")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(df), batch_size):
        batch_df = df.iloc[i:i + batch_size]
        batch_id  = f"BATCH_Y{year}_{task_name}_B{i//batch_size}"
        batch_df.to_csv(output_dir / f"{batch_id}.csv", index=False, encoding="utf-8-sig")
        print(f"✅ 生成 {batch_id}.csv")
