"""
organize_datasets.py
────────────────────────────────────────────────────────────────────────────
Creates the 1K / 5K / 10K folder structure for geometry datasets.

Each folder gets:
  1. Raw extracted parquet  (DeepVision_Geometry_{N}.parquet)
  2. Ready-to-Translate parquet  (DeepBanglaMath_{N}_Ready_to_Translate.parquet)
     → original + empty bangla_question, translation_status, bangla_prompt cols
  3. chunks/ folder with Excel files  (3-column: row_id, english_question, bangla_question)
  4. Final translated parquet (for 1K only, since we have the translations)

Source: dataset/DeepVision-103K/math-77k.parquet  (geometry subset = 58 735 rows)
"""

import os
import shutil
import pandas as pd
import numpy as np

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
SOURCE_10K  = os.path.join(DATASET_DIR, "10K Geometry", "DeepVision_Geometry_10K.parquet")

# ── Helpers ───────────────────────────────────────────────────────────────────

def create_ready_to_translate(df: pd.DataFrame) -> pd.DataFrame:
    """Add empty translation columns to raw data → ready-to-translate version."""
    ready = df.copy()
    ready["bangla_question"]     = pd.array([""] * len(ready), dtype="string")
    ready["translation_status"]  = pd.array(["pending"] * len(ready), dtype="string")
    ready["bangla_prompt"]       = np.empty(len(ready), dtype=object)
    for i in range(len(ready)):
        ready.iat[i, ready.columns.get_loc("bangla_prompt")] = None
    return ready


def export_chunks(df: pd.DataFrame, chunks_dir: str, chunk_size: int = 200):
    """Export 3-column CSV chunks for manual translation."""
    os.makedirs(chunks_dir, exist_ok=True)
    q_col = "question"
    n_chunks = (len(df) + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        start = i * chunk_size
        end   = min(start + chunk_size, len(df))
        chunk = df.iloc[start:end]

        out_path = os.path.join(chunks_dir, f"chunk_{i+1:02d}_rows_{start}-{end-1}.csv")
        pd.DataFrame({
            "row_id":           range(start, end),
            "english_question": chunk[q_col].values,
            "bangla_question":  "",
        }).to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"    Exported {n_chunks} chunks ({chunk_size} rows each) → {chunks_dir}")


# ── Load source ───────────────────────────────────────────────────────────────
print("Loading source: DeepVision_Geometry_10K.parquet ...")
full_df = pd.read_parquet(SOURCE_10K).reset_index(drop=True)
print(f"  Total rows: {len(full_df):,}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 1K GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("  1K GEOMETRY")
print("=" * 60)

dir_1k = os.path.join(DATASET_DIR, "1K Geometry")
os.makedirs(dir_1k, exist_ok=True)

# 1a. Raw 1K parquet (first 1000 rows)
df_1k = full_df.iloc[:1000].copy().reset_index(drop=True)
raw_1k_path = os.path.join(dir_1k, "DeepVision_Geometry_1K.parquet")
df_1k.to_parquet(raw_1k_path, index=False)
print(f"  Raw: {raw_1k_path}  ({len(df_1k)} rows)")

# 1b. Ready-to-translate 1K
ready_1k = create_ready_to_translate(df_1k)
ready_1k_path = os.path.join(dir_1k, "DeepBanglaMath_1K_Ready_to_Translate.parquet")
ready_1k.to_parquet(ready_1k_path, index=False)
print(f"  Ready: {ready_1k_path}  ({len(ready_1k)} rows)")

# 1c. Copy translated Excel chunks from 10K/chunks/ → 1K/chunks/
src_chunks = os.path.join(DATASET_DIR, "10K Geometry", "chunks")
dst_chunks_1k = os.path.join(dir_1k, "chunks")
os.makedirs(dst_chunks_1k, exist_ok=True)

xlsx_files = [f for f in os.listdir(src_chunks)
              if f.endswith(".csv.xlsx") and int(f.split("rows_")[1].split("-")[0]) < 1000]
for f in sorted(xlsx_files):
    shutil.copy2(os.path.join(src_chunks, f), os.path.join(dst_chunks_1k, f))
    print(f"    Copied {f} → 1K/chunks/")

# 1d. Copy final translated parquet
src_final = os.path.join(DATASET_DIR, "10K Geometry", "DeepBanglaMath_1000_translated.parquet")
dst_final_1k = os.path.join(dir_1k, "DeepBanglaMath_1K_translated.parquet")
shutil.copy2(src_final, dst_final_1k)
print(f"  Final: {dst_final_1k}")


# ══════════════════════════════════════════════════════════════════════════════
# 5K GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  5K GEOMETRY")
print("=" * 60)

dir_5k = os.path.join(DATASET_DIR, "5K Geometry")
os.makedirs(dir_5k, exist_ok=True)

# 5a. Raw 5K parquet (first 5000 rows)
df_5k = full_df.iloc[:5000].copy().reset_index(drop=True)
raw_5k_path = os.path.join(dir_5k, "DeepVision_Geometry_5K.parquet")
df_5k.to_parquet(raw_5k_path, index=False)
print(f"  Raw: {raw_5k_path}  ({len(df_5k)} rows)")

# 5b. Ready-to-translate 5K
ready_5k = create_ready_to_translate(df_5k)
ready_5k_path = os.path.join(dir_5k, "DeepBanglaMath_5K_Ready_to_Translate.parquet")
ready_5k.to_parquet(ready_5k_path, index=False)
print(f"  Ready: {ready_5k_path}  ({len(ready_5k)} rows)")

# 5c. Export chunks for translation
chunks_5k = os.path.join(dir_5k, "chunks")
export_chunks(df_5k, chunks_5k, chunk_size=200)


# ══════════════════════════════════════════════════════════════════════════════
# 10K GEOMETRY  (already exists, just verify & add missing pieces)
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  10K GEOMETRY (verify existing)")
print("=" * 60)

dir_10k = os.path.join(DATASET_DIR, "10K Geometry")

# Verify raw
raw_10k = os.path.join(dir_10k, "DeepVision_Geometry_10K.parquet")
print(f"  Raw: {raw_10k}  (exists={os.path.exists(raw_10k)})")

# Verify ready
ready_10k = os.path.join(dir_10k, "DeepBanglaMath_10K_Ready_to_Translate.parquet")
print(f"  Ready: {ready_10k}  (exists={os.path.exists(ready_10k)})")

# Verify chunks
chunks_10k = os.path.join(dir_10k, "chunks")
chunk_files = os.listdir(chunks_10k)
print(f"  Chunks: {len(chunk_files)} files in {chunks_10k}")

# Verify final (1000 translated)
final_10k = os.path.join(dir_10k, "DeepBanglaMath_1000_translated.parquet")
print(f"  Final (1K translated): {final_10k}  (exists={os.path.exists(final_10k)})")


# ══════════════════════════════════════════════════════════════════════════════
# CLEANUP: remove old/duplicate files
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  CLEANUP")
print("=" * 60)

# Remove old "1k Geometry" folder (lowercase, duplicate of new "1K Geometry")
# NOTE: On Windows, paths are case-insensitive so "1k" == "1K".
#       We skip this step to avoid deleting the folder we just created.
# old_1k = os.path.join(DATASET_DIR, "1k Geometry")
# if os.path.exists(old_1k) and os.path.realpath(old_1k) != os.path.realpath(dir_1k):
#     shutil.rmtree(old_1k)

# Remove old root-level parquet files
for old_file in ["DeepBanglaMath_100_READY.parquet", "DeepBanglaMath_1K_READY.parquet"]:
    p = os.path.join(BASE_DIR, old_file)
    if os.path.exists(p):
        os.remove(p)
        print(f"  Removed: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 60}")
print("  FINAL FOLDER STRUCTURE")
print("=" * 60)

for root, dirs, files in os.walk(DATASET_DIR):
    # Skip DeepVision-103K (source, too large)
    if "DeepVision-103K" in root:
        continue
    level = root.replace(DATASET_DIR, "").count(os.sep)
    indent = "  " * level
    folder = os.path.basename(root) or "dataset/"
    print(f"{indent}{folder}/")
    for f in sorted(files):
        size_mb = os.path.getsize(os.path.join(root, f)) / (1024 * 1024)
        print(f"{indent}  {f:55s}  {size_mb:8.2f} MB")

print("\n✅ Done!")
