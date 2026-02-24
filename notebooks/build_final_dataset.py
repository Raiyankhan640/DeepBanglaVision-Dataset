"""
build_final_dataset.py
───────────────────────────────────────────────────────────────────────────────
Merges 5 translated CSV chunks with the original Parquet file and produces
a final Bangla math dataset in Parquet format.

Steps:
  1. Read chunk_01 – chunk_05 CSVs (1000 translated rows)
  2. Concatenate into one DataFrame, sorted by row_id
  3. Load original DeepVision_Geometry_10K.parquet
  4. Merge translated questions into the Parquet data
  5. Build bangla_assistant (templated answer) and messages (ChatML) columns
  6. Save as DeepBanglaMath_1000_translated.parquet

Usage:
  .venv\\Scripts\\python.exe notebooks/build_final_dataset.py
"""

import os
import json
import pandas as pd
from pprint import pprint

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_DIR  = os.path.join(BASE_DIR, "dataset", "10K Geometry", "chunks")
PARQUET_SRC = os.path.join(BASE_DIR, "dataset", "10K Geometry",
                           "DeepVision_Geometry_10K.parquet")
OUTPUT_FILE = os.path.join(BASE_DIR, "dataset", "10K Geometry",
                           "DeepBanglaMath_1000_translated.parquet")

# The 5 translated chunk files to merge (rows 0-999)
# NOTE: files are .csv.xlsx (Excel format) — saved from spreadsheet app
CHUNK_FILES = [
    "chunk_01_rows_0-199.csv.xlsx",
    "chunk_02_rows_200-399.csv.xlsx",
    "chunk_03_rows_400-599.csv.xlsx",
    "chunk_04_rows_600-799.csv.xlsx",
    "chunk_05_rows_800-999.csv.xlsx",
]


# ── STEP 1 & 2: Read and concatenate all CSV chunks ──────────────────────────
def load_and_concat_chunks() -> pd.DataFrame:
    """Read all 5 CSV chunk files and return a single concatenated DataFrame."""
    frames = []
    for filename in CHUNK_FILES:
        filepath = os.path.join(CHUNKS_DIR, filename)
        print(f"  Reading {filename} ... ", end="", flush=True)
        try:
            # Read Excel (.xlsx) or CSV depending on extension
            if filepath.endswith(".xlsx"):
                chunk = pd.read_excel(filepath, dtype=str)
            else:
                try:
                    chunk = pd.read_csv(filepath, encoding="utf-8-sig", dtype=str)
                except UnicodeDecodeError:
                    chunk = pd.read_csv(filepath, encoding="utf-8", dtype=str)

            # Remove any rows that are duplicate headers (header repeated in data)
            chunk = chunk[chunk["row_id"] != "row_id"]

            # Convert row_id to int
            chunk["row_id"] = chunk["row_id"].astype(int)

            print(f"{len(chunk)} rows ✅")
            frames.append(chunk)

        except Exception as e:
            print(f"❌ Error: {e}")
            raise

    # Concatenate and sort by row_id
    combined = pd.concat(frames, ignore_index=True)
    combined.sort_values("row_id", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    # Drop duplicates on row_id (keep first occurrence)
    before = len(combined)
    combined.drop_duplicates(subset="row_id", keep="first", inplace=True)
    if len(combined) < before:
        print(f"  ⚠️  Dropped {before - len(combined)} duplicate row_id(s)")

    print(f"\n📦 Combined CSV shape: {combined.shape}")
    print(f"   row_id range: {combined['row_id'].min()} – {combined['row_id'].max()}")
    return combined


# ── STEP 3–6: Load Parquet, merge, build columns ─────────────────────────────
def build_dataset(csv_df: pd.DataFrame) -> pd.DataFrame:
    """Load Parquet, merge with translations, build assistant & messages cols."""

    # --- Step 4: Load original Parquet ---
    print(f"\n📂 Loading Parquet: {os.path.basename(PARQUET_SRC)} ... ", end="", flush=True)
    try:
        pq_df = pd.read_parquet(PARQUET_SRC)
        print(f"{pq_df.shape[0]} rows ✅")
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

    # --- Step 5: Reset index → sequential row_id (0, 1, 2, ...) ---
    pq_df = pq_df.reset_index(drop=True)
    pq_df["row_id"] = pq_df.index.astype(int)
    print(f"   Parquet row_id range: {pq_df['row_id'].min()} – {pq_df['row_id'].max()}")

    # Keep only the first 1000 rows (matching our translated range)
    pq_df = pq_df[pq_df["row_id"].isin(csv_df["row_id"])].copy()
    print(f"   Filtered to {len(pq_df)} rows (matching CSV row_ids)")

    # --- Step 6: Left join Parquet with translated CSV on row_id ---
    merged = pq_df.merge(
        csv_df[["row_id", "bangla_question"]],
        on="row_id",
        how="left",
    )
    print(f"\n🔗 Merged shape: {merged.shape}")

    # --- Step 7: Fill NaN bangla_question with original question ---
    null_count = merged["bangla_question"].isna().sum()
    if null_count > 0:
        print(f"   ⚠️  {null_count} rows have NaN bangla_question → filling with 'question'")
        merged["bangla_question"] = merged["bangla_question"].fillna(merged["question"])
    else:
        print("   ✅ No NaN in bangla_question")

    # --- Step 8: Create bangla_assistant column ---
    print("\n🛠️  Building 'bangla_assistant' column ...")

    def make_bangla_assistant(row):
        """Build the templated Bangla assistant response."""
        try:
            gt = row["reward_model"]["ground_truth"]
        except (TypeError, KeyError):
            gt = "N/A"

        return (
            "চিত্রটি ভালোভাবে দেখে ধাপে ধাপে সমাধান করি।\n\n"
            "প্রথমে চিত্রের সব তথ্য চিহ্নিত করি...\n"
            "[সংক্ষিপ্ত যুক্তি]\n\n"
            f"সুতরাং, সঠিক উত্তর \\boxed{{{gt}}}"
        )

    merged["bangla_assistant"] = merged.apply(make_bangla_assistant, axis=1)

    # --- Step 9: Create messages column (ChatML format) ---
    print("🛠️  Building 'messages' column (ChatML) ...")

    def make_messages(row):
        """Build ChatML-format messages list."""
        # prompt is a numpy array/list of dicts; prompt[0] = system message
        system_msg = row["prompt"][0]  # {"role": "system", "content": "..."}

        user_msg = {
            "role": "user",
            "content": f"<image>{row['bangla_question']}",
        }

        assistant_msg = {
            "role": "assistant",
            "content": row["bangla_assistant"],
        }

        return [system_msg, user_msg, assistant_msg]

    merged["messages"] = merged.apply(make_messages, axis=1)

    return merged


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("  BUILD FINAL BANGLA MATH DATASET")
    print("=" * 65)

    # Step 1-2: Load and concatenate CSV chunks
    print("\n── Step 1-2: Loading CSV chunks ──")
    csv_df = load_and_concat_chunks()

    # Steps 3-9: Build the dataset
    print("\n── Steps 3-9: Building dataset ──")
    final_df = build_dataset(csv_df)

    # Step 10: Save final Parquet
    print(f"\n💾 Saving to: {os.path.basename(OUTPUT_FILE)} ... ", end="", flush=True)
    final_df.to_parquet(OUTPUT_FILE, index=False)
    print("✅")

    # Step 11: Print shape and first 3 messages
    print(f"\n{'=' * 65}")
    print(f"📊 Final DataFrame shape: {final_df.shape}")
    print(f"   Columns: {final_df.columns.tolist()}")
    print(f"\n── First 3 'messages' values ──\n")
    for i in range(min(3, len(final_df))):
        print(f"--- Row {final_df.iloc[i]['row_id']} ---")
        pprint(final_df.iloc[i]["messages"], width=100)
        print()

    print("✅ Done!")


if __name__ == "__main__":
    main()
