from pathlib import Path
import pandas as pd, joblib

def add_prob_column(csv_in: Path, model_pkl: Path, csv_out: Path, abs_col="abstract"):
    df   = pd.read_csv(csv_in)
    mdl  = joblib.load(model_pkl)
    df   = df[df[abs_col].notna() & (df[abs_col].str.strip() != "")]
    df["aging_probability"] = mdl.predict_proba(df[abs_col])[:,1]
    df.to_csv(csv_out, index=False, encoding="utf-8-sig")
    return df
