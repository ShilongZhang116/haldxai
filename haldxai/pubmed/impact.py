import pandas as pd
from impact_factor.core import Factor

def annotate_journals_with_if(df):
    fa = Factor()
    df['issn'] = df['issn'].astype(str)

    # 1. ISSN 查询
    unique_issn = df['issn'].dropna().drop_duplicates()
    issn_df = pd.DataFrame({"issn": unique_issn})
    issn_df["journal_info"] = issn_df["issn"].apply(fa.search)

    df["journal_info"] = df["issn"].map(issn_df.set_index("issn")["journal_info"])

    # 2. 期刊名补查
    missing_df = df[df["journal_info"].isnull()]
    titles = missing_df["journal_full_title"].dropna().drop_duplicates()
    title_df = pd.DataFrame({"journal_full_title": titles})
    title_df["journal_info"] = title_df["journal_full_title"].apply(fa.search)

    df.loc[missing_df.index, "journal_info"] = df.loc[missing_df.index, "journal_full_title"].map(
        title_df.set_index("journal_full_title")["journal_info"]
    )

    # 3. 删除仍然没有结果的
    df = df.dropna(subset=["journal_info"]).copy()

    # 4. 展开 info
    df["article_info_dict"] = df["journal_info"].apply(
        lambda x: x[0] if isinstance(x, list) and x else {}
    )
    expanded = df["article_info_dict"].apply(pd.Series)
    return pd.concat([df.drop(columns=["journal_info", "article_info_dict"]), expanded], axis=1)
