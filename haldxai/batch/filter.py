from pathlib import Path
import pandas as pd, joblib, os

# ========== 公共小工具 ==========
def append_aging_prob(df: pd.DataFrame, model_path: Path, abs_col: str = "abstract") -> pd.DataFrame:
    """
    若 df 中无 aging_probability，则载入模型并计算。
    * 模型必须支持 .predict_proba
    """
    if "aging_probability" in df.columns:
        return df                         # 已有，直接返回

    if not model_path.exists():
        raise FileNotFoundError(f"模型不存在: {model_path}")

    model = joblib.load(model_path)
    mask  = df[abs_col].notna() & (df[abs_col].str.strip() != "")
    df.loc[mask, "aging_probability"] = model.predict_proba(df.loc[mask, abs_col])[:, 1]
    return df


# ========== 单独过滤器函数 ==========
def filter_q1_if10(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        (df["jcr"] == "Q1") &
        (df["factor"] > 10) &
        df["abstract"].notna() &
        (df["abstract"].str.strip() != "")
    ]

def filter_only_abstract(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["abstract"].notna() & (df["abstract"].str.strip() != "")]


# ========== 动态生成“按模型打分 + Top-K”过滤器 ==========
def make_filter_topk_by_model(model_path: Path, top_n: int = 30_000):
    """
    返回一个闭包函数：先用模型计算概率，再取 Top-N。
    """
    def _filter(df: pd.DataFrame) -> pd.DataFrame:
        df = append_aging_prob(df.copy(), model_path)
        df = df[df["abstract"].notna() & (df["abstract"].str.strip() != "")]
        return df.nlargest(top_n, "aging_probability").reset_index(drop=True)
    return _filter


# ========== 过滤器注册表 ==========
_MODEL_DIR = Path(os.getenv("HALDXAI_MODEL_DIR", Path(__file__).resolve().parents[2]/"models"))
AGING_MODEL_PATH = _MODEL_DIR / "aging_classifier_tfidf_lr_v1" / "model.pkl"

_FILTER_REGISTRY = {
    "JCRQ1-IF10":      filter_q1_if10,
    "ONLY_ABSTRACT":   filter_only_abstract,
    "AgingRelated":    make_filter_topk_by_model(AGING_MODEL_PATH, top_n=5000),
}

def get_filter(task_name: str):
    if task_name not in _FILTER_REGISTRY:
        raise ValueError(f"❌ 未注册的 Task: {task_name}")
    return _FILTER_REGISTRY[task_name]
