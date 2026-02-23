import pandas as pd
import numpy as np
import os

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
SRC_FILE_10K  = os.path.join(BASE_DIR, "DeepVision_Geometry_10K.parquet")
SRC_FILE_1K   = os.path.join(BASE_DIR, "DeepVision_Geometry_Pilot_1K.parquet")
OUT_FILE_10K  = os.path.join(BASE_DIR, "DeepBanglaMath_10K_READY.parquet")
OUT_FILE_1K   = os.path.join(BASE_DIR, "DeepBanglaMath_1K_READY.parquet")
TRANS_DIR     = os.path.join(BASE_DIR, "translations")
os.makedirs(TRANS_DIR, exist_ok=True)

# ── Helper: add Bangla columns ────────────────────────────────────────────────
def build_bangla_prompt(row):
    messages  = list(row["prompt"])
    system    = messages[0]
    user_text = row["question"]   # English fallback until translated
    return [system, {"role": "user", "content": f"<image>{user_text}"}]

def prepare_bangla_dataset(src_path, out_path, label):
    print(f"\n[{label}] Loading: {src_path}")
    df = pd.read_parquet(src_path)
    print(f"[{label}] Loaded {len(df):,} rows × {len(df.columns)} columns")

    df.insert(df.columns.get_loc("question") + 1, "bangla_question", "")
    df["translation_status"] = "pending"
    df["bangla_prompt"]      = df.apply(build_bangla_prompt, axis=1)

    df.to_parquet(out_path, index=False)
    print(f"[{label}] ✅ Saved → {out_path}")
    print(f"[{label}]    Columns: {df.columns.tolist()}")
    return df

# ── Build both ready files ────────────────────────────────────────────────────
df_10k = prepare_bangla_dataset(SRC_FILE_10K, OUT_FILE_10K, "10K")
df_1k  = prepare_bangla_dataset(SRC_FILE_1K,  OUT_FILE_1K,  "1K")

# ── Export first translation batch from 1K (rows 0–99) ───────────────────────
BATCH_SIZE  = 100
batch_start = 0
batch_end   = batch_start + BATCH_SIZE
batch_num   = 1

batch = df_1k.iloc[batch_start:batch_end].copy()

trans_csv = os.path.join(
    TRANS_DIR,
    f"batch_{batch_num:02d}_rows_{batch_start}-{batch_end - 1}.csv"
)
trans_df = pd.DataFrame({
    "row_id":           batch.index,
    "english_question": batch["question"].values,
    "bangla_question":  ""
})
trans_df.to_csv(trans_csv, index=False)

print(f"\n[Batch] ✅ Translation CSV saved → {trans_csv}")
print(f"[Batch]    {len(batch)} rows — fill Column C (bangla_question) and send back.")
print("\nAll done!")