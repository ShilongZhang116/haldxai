# haldxai/pubmed/fetch.py
import io
import os
import time
import yaml
import pandas as pd
from Bio import Entrez, Medline
from urllib.error import HTTPError, URLError
from haldxai.pubmed.process import process_record, merge_results


def fetch_pubmed_data(query, email, summary_file, api_key=None, retmax=None, batch_size=500):
    """
    主函数：使用 Entrez API 获取 PubMed 文献数据并存为 csv，带 checkpoint。
    """
    Entrez.email = email
    if api_key:
        Entrez.api_key = api_key

    retmax = retmax or 100000

    search_handle = Entrez.esearch(
        db="pubmed", term=query, usehistory="y", retmax=retmax
    )
    search_result = Entrez.read(search_handle)
    search_handle.close()

    webenv = search_result["WebEnv"]
    query_key = search_result["QueryKey"]
    total_count = int(search_result["Count"])

    if retmax > total_count:
        print(f"🔍 找到 {total_count} 篇文献，计划获取 {total_count} 篇")
    else:
        print(f"🔍 找到 {total_count} 篇文献，计划获取 {retmax} 篇")

    downloaded_pmids = set()
    checkpoint_df = pd.DataFrame()

    if os.path.exists(summary_file):
        try:
            checkpoint_df = pd.read_csv(summary_file, dtype=str)
            downloaded_pmids = set(str(int(float(pmid))) for pmid in checkpoint_df["pmid"].dropna())
            print(f"✅ 加载 Checkpoint，已下载 {len(downloaded_pmids)} 篇")
        except Exception as e:
            print(f"⚠️ Checkpoint 读取失败: {e}")

    all_pmids = set(str(pmid) for pmid in search_result["IdList"])
    remaining_pmids = list(all_pmids - downloaded_pmids)


    if not remaining_pmids:
        print("🎉 所有文献已下载")
        return checkpoint_df

    new_results = []
    for i in range(0, len(remaining_pmids), batch_size):
        batch_pmids = remaining_pmids[i:i+batch_size]
        try:
            fetch_handle = Entrez.efetch(
                db="pubmed", id=",".join(batch_pmids),
                rettype="medline", retmode="text"
            )
            data = fetch_handle.read()
            fetch_handle.close()

            batch = [process_record(rec) for rec in Medline.parse(io.StringIO(data))]
            df_batch = pd.DataFrame(batch)
            df_batch.to_csv(summary_file, mode='a', header=not os.path.exists(summary_file),
                            index=False, encoding='utf-8-sig')
            new_results.extend(batch)
            print(f"✅ 获取 {i+1}-{i+len(batch)} 条")

            time.sleep(1)
        except Exception as e:
            print(f"⛔ 错误：{e}")
            continue

    return merge_results(checkpoint_df, new_results)


def generate_query_with_time(query, start_date, end_date):
    return f"({query}) AND ({start_date}[Date - Publication] : {end_date}[Date - Publication])"
