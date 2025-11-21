import pandas as pd
from datetime import datetime


def extract_publication_year(pub_date_str):
    try:
        if not pub_date_str:
            return None
        clean_date = pub_date_str.replace('-', ' ').replace('/', ' ').split()[0]
        year = int(clean_date[:4]) if len(clean_date) >= 4 else None
        return year if 1900 <= year <= datetime.now().year else None
    except:
        return None


def save_yearly_data(df, base_path):
    df['year'] = df['pub_date'].apply(extract_publication_year)
    valid_df = df.dropna(subset=['year'])
    invalid_count = len(df) - len(valid_df)
    if invalid_count > 0:
        print(f"⚠️ 发现 {invalid_count} 条记录存在日期格式问题")

    for year in sorted(valid_df['year'].astype(int).unique()):
        year_df = valid_df[valid_df['year'] == year]
        year_df.to_csv(f"{base_path}_Y{year}.csv", index=False, encoding='utf-8-sig')
        print(f"📁 已保存 {year} 年文献数据（{len(year_df)} 篇）")

    return valid_df
