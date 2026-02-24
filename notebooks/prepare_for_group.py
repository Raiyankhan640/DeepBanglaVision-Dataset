import pandas as pd
import numpy as np
import json
import os

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
RAW_77K       = os.path.join(BASE_DIR, "dataset", "DeepVision-103K", "math-77k.parquet")
SRC_FILE_10K  = os.path.join(BASE_DIR, "DeepVision_Geometry_10K.parquet")
SRC_FILE_1K   = os.path.join(BASE_DIR, "dataset", "1k Geometry", "DeepVision_Geometry_1K.parquet")
OUT_FILE_10K  = os.path.join(BASE_DIR, "DeepBanglaMath_10K_READY.parquet")
OUT_FILE_1K   = os.path.join(BASE_DIR, "DeepBanglaMath_1K_READY.parquet")
TRANS_DIR     = os.path.join(BASE_DIR, "translations")
CHUNKS_DIR    = os.path.join(TRANS_DIR, "chunks")
os.makedirs(TRANS_DIR, exist_ok=True)
os.makedirs(CHUNKS_DIR, exist_ok=True)

# ── Step 0: Generate DeepVision_Geometry_10K.parquet if missing ───────────────
def build_geometry_10k():
    if os.path.exists(SRC_FILE_10K):
        print(f"[10K-filter] Already exists, skipping filter step.")
        return

    print(f"[10K-filter] Filtering geometry rows from {RAW_77K} ...")
    df = pd.read_parquet(RAW_77K)
    print(f"[10K-filter] Loaded {len(df):,} rows")

    def is_geometry(domain_str):
        try:
            d = json.loads(domain_str)
            if d.get("Level_1") == "Geometry":
                return True
            keywords = ["Triangle", "Circle", "Polygon", "Geometry", "Angle", "Plane"]
            return any(any(kw in k for kw in keywords) for k in d.get("Knowledge", []))
        except Exception:
            return False

    geo_df = df[df["domain-anno"].apply(is_geometry)].reset_index(drop=True)
    geo_df.to_parquet(SRC_FILE_10K, index=False)
    print(f"[10K-filter] ✅ {len(geo_df):,} geometry rows saved → {SRC_FILE_10K}")

build_geometry_10k()

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

# ── Export 3 × 200-row chunks from 10K READY (chronological order) ────────────
CHUNK_SIZE   = 200
NUM_CHUNKS   = 3

print(f"\n[Chunks] Exporting {NUM_CHUNKS} chunks of {CHUNK_SIZE} rows from 10K READY ...")
df_10k_clean = df_10k.reset_index(drop=True)   # ensure 0-based sequential index

for i in range(NUM_CHUNKS):
    row_start = i * CHUNK_SIZE
    row_end   = row_start + CHUNK_SIZE
    chunk     = df_10k_clean.iloc[row_start:row_end].copy()

    chunk_csv = os.path.join(
        CHUNKS_DIR,
        f"chunk_{i+1:02d}_rows_{row_start}-{row_end - 1}.csv"
    )
    chunk_df = pd.DataFrame({
        "row_id":           chunk.index,
        "english_question": chunk["question"].values,
        "bangla_question":  ""
    })
    chunk_df.to_csv(chunk_csv, index=False)
    print(f"[Chunks]   chunk {i+1:02d}: rows {row_start}–{row_end - 1} → {chunk_csv}")

print(f"\n[Chunks] ✅ All {NUM_CHUNKS} chunks saved to {CHUNKS_DIR}")
print("\nAll done!")