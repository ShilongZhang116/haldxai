from pathlib import Path
import joblib, json, datetime, getpass, platform

def save_model(model, model_name: str, project_root: Path, meta: dict | None = None):
    """
    将训练好的模型保存到  <project_root>/models/<model_name>/model.pkl
    同时自动生成 meta.json 记录关键信息。
    """
    model_dir = project_root / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # 1) 保存二进制模型
    model_file = model_dir / "model.pkl"
    joblib.dump(model, model_file)
    print(f"✅ 模型已保存: {model_file}")

    # 2) 生成 / 更新 meta.json
    meta_file = model_dir / "meta.json"
    default_meta = dict(
        model_name   = model_name,
        saved_time   = datetime.datetime.now().isoformat(timespec="seconds"),
        author       = getpass.getuser(),
        platform     = platform.platform(),
    )
    if meta:
        default_meta.update(meta)

    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(default_meta, f, indent=2, ensure_ascii=False)
    print(f"📝 元数据已保存: {meta_file}")
