"""
export_chunks.py
Exports 3 translation-ready CSVs (200 rows each) from the existing 10K READY parquet.

Source:  dataset/10K Geometry/DeepBanglaMath_10K_Ready_to_Translate.parquet
Output:  dataset/10K Geometry/chunks/
  chunk_01_rows_0-199.csv
  chunk_02_rows_200-399.csv
  chunk_03_rows_400-599.csv
"""

import pandas as pd
import os

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
READY_10K  = os.path.join(BASE_DIR, "dataset", "10K Geometry", "DeepBanglaMath_10K_Ready_to_Translate.parquet")
CHUNKS_DIR = os.path.join(BASE_DIR, "dataset", "10K Geometry", "chunks")
os.makedirs(CHUNKS_DIR, exist_ok=True)

# ── Load READY parquet ─────────────────────────────────────────────────────────
print(f"Loading: {READY_10K}")
df = pd.read_parquet(READY_10K).reset_index(drop=True)
print(f"Loaded {len(df):,} rows × {len(df.columns)} columns")
print(f"Columns: {df.columns.tolist()}\n")

# ── Export 3 × 200 chronological chunks ───────────────────────────────────────
CHUNK_SIZE = 200
NUM_CHUNKS = 3

q_col = "question" if "question" in df.columns else df.columns[0]

for i in range(NUM_CHUNKS):
    start = i * CHUNK_SIZE
    end   = start + CHUNK_SIZE
    chunk = df.iloc[start:end]

    out_path = os.path.join(CHUNKS_DIR, f"chunk_{i+1:02d}_rows_{start}-{end-1}.csv")
    pd.DataFrame({
        "row_id":           chunk.index,
        "english_question": chunk[q_col].values,
        "bangla_question":  "",
    }).to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"chunk {i+1:02d}: rows {start}–{end-1}  →  {out_path}")

print(f"\n✅ Done. {NUM_CHUNKS} CSVs saved to: {CHUNKS_DIR}")
