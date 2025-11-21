from pathlib import Path
import pandas as pd, joblib, numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, classification_report

__all__ = ["train_model"]

def load_pos_neg(pos_csv: Path, neg_csv: Path, aging_journals: list[str], neg_ratio: int):
    df_pos = pd.read_csv(pos_csv)
    df_neg = pd.read_csv(neg_csv)

    df_pos = df_pos[df_pos["journal_full_title"].isin(aging_journals)].copy()
    df_pos["label"] = 1
    df_neg["label"] = 0

    df_neg = df_neg.sample(n=min(len(df_neg), len(df_pos)*neg_ratio), random_state=42)
    df_all = pd.concat([df_pos, df_neg], ignore_index=True)
    df_all = df_all[df_all["abstract"].notna() & (df_all["abstract"].str.strip() != "")]
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_all["abstract"], df_all["label"]

def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
        ("clf",   LogisticRegression(class_weight="balanced", max_iter=1000))
    ])

def cross_validate(X, y, n_split=5):
    skf = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=42)
    aucs, last_fold = [], None
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        model = build_pipeline()
        model.fit(X.iloc[tr], y.iloc[tr])
        prob  = model.predict_proba(X.iloc[te])[:,1]
        aucs.append(roc_auc_score(y.iloc[te], prob))
        print(f"✅ fold {fold} AUC={aucs[-1]:.4f}")
        last_fold = (model, X.iloc[te], y.iloc[te])      # for viz
    print(f"🌟 mean AUC = {np.mean(aucs):.4f}")
    return aucs, last_fold

def train_model(
    pos_csv: Path,
    neg_csv: Path,
    model_out: Path,
    aging_journals:list[str],
    neg_ratio:int=3,
    show_cv: bool = True
):
    X, y = load_pos_neg(pos_csv, neg_csv, aging_journals, neg_ratio)
    aucs, last_fold = cross_validate(X, y) if show_cv else ([], None)

    final_model = build_pipeline().fit(X, y)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_out)
    print(f"💾 saved: {model_out}")

    return {"aucs": aucs, "last_fold": last_fold, "model": final_model}
