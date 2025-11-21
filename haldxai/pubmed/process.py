# haldxai/pubmed/process.py
import re
import pandas as pd
from datetime import datetime, timedelta

def process_record(rec):
    return {
        "pmid": rec.get('PMID', ''),
        "title": rec.get('TI', '').strip('.'),
        "abstract": rec.get('AB', ''),
        "journal_full_title": rec.get('JT', ''),
        "journal_abbr": rec.get('TA', ''),
        "issn": parse_issn(rec.get('IS', '')),
        "nlm_id": rec.get('JID', ''),
        "pub_date": format_pub_date(rec.get('DP', '')),
        "authors": "; ".join(rec.get('AU', [])),
        "pub_types": ", ".join(rec.get('PT', []))
    }

def parse_issn(issn_str):
    return issn_str.split()[0] if '-' in issn_str else issn_str

def format_pub_date(date_str):
    parts = re.findall(r'\d+', date_str)
    if len(parts) >= 1:
        year = parts[0]
        month = parts[1] if len(parts) >= 2 else '01'
        return f"{year}-{month.zfill(2)}"
    return ''

def generate_monthly_ranges(start_year, end_year):
    ranges = []
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            start = f"{year}/{month:02d}/01"
            next_month = datetime(year, month, 1) + timedelta(days=31)
            end = f"{next_month.year}/{next_month.month:02d}/01"
            ranges.append((start, end))
    return ranges

def merge_results(old_df, new_results):
    new_df = pd.DataFrame(new_results)
    if old_df.empty:
        return new_df
    combined = pd.concat([old_df, new_df], ignore_index=True)
    combined.drop_duplicates(subset=["pmid"], inplace=True)
    return combined
