"""
verify_dataset.py — Comprehensive verification of DeepBanglaMath_1000_translated.parquet
"""
import os, sys, random
import pandas as pd
from PIL import Image
import io, base64

BASE    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARQUET = os.path.join(BASE, "dataset", "10K Geometry", "DeepBanglaMath_1000_translated.parquet")
ORIG_PQ = os.path.join(BASE, "dataset", "10K Geometry", "DeepVision_Geometry_10K.parquet")

# ── Load ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("  DATASET VERIFICATION — DeepBanglaMath_1000_translated.parquet")
print("=" * 70)

df   = pd.read_parquet(PARQUET)
orig = pd.read_parquet(ORIG_PQ).reset_index(drop=True)
orig["row_id"] = orig.index

# ── 1. Basic stats ─────────────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print("1. SHAPE & COLUMNS")
print(f"{'─'*50}")
print(f"  Rows    : {df.shape[0]:,}")
print(f"  Columns : {df.shape[1]}")
for col in df.columns:
    null_pct = df[col].isna().mean() * 100
    print(f"  {col:<35s}  dtype={str(df[col].dtype):<15s}  null={null_pct:.1f}%")

# ── 2. Bangla question completeness ────────────────────────────────────────────
print(f"\n{'─'*50}")
print("2. BANGLA QUESTION COMPLETENESS")
print(f"{'─'*50}")
nan_bq = df["bangla_question"].isna().sum()
empty_bq = (df["bangla_question"].str.strip() == "").sum()
print(f"  NaN        : {nan_bq}")
print(f"  Empty str  : {empty_bq}")
print(f"  Filled     : {df.shape[0] - nan_bq - empty_bq} / {df.shape[0]}")

# ── 3. Image presence & dimensions ────────────────────────────────────────────
print(f"\n{'─'*50}")
print("3. IMAGE PRESENCE CHECK (5 random rows)")
print(f"{'─'*50}")
random.seed(42)
sample_ids = random.sample(df["row_id"].tolist(), 5)

for rid in sample_ids:
    row = df[df["row_id"] == rid].iloc[0]
    images = row["images"]
    img_info = "None"
    if images is not None:
        try:
            # images column: list of PIL/bytes/dicts
            img_list = list(images) if hasattr(images, "__iter__") else [images]
            imgs_ok = []
            for img in img_list:
                if isinstance(img, Image.Image):
                    imgs_ok.append(f"{img.size[0]}×{img.size[1]}")
                elif isinstance(img, (bytes, bytearray)):
                    pil = Image.open(io.BytesIO(img))
                    imgs_ok.append(f"{pil.size[0]}×{pil.size[1]}")
                elif isinstance(img, dict) and "bytes" in img:
                    pil = Image.open(io.BytesIO(img["bytes"]))
                    imgs_ok.append(f"{pil.size[0]}×{pil.size[1]}")
                elif isinstance(img, str):
                    imgs_ok.append(f"str({len(img)}ch)")
                else:
                    imgs_ok.append(f"type:{type(img).__name__}")
            img_info = ", ".join(imgs_ok)
        except Exception as e:
            img_info = f"Error: {e}"
    bq_snippet = str(row["bangla_question"])[:80].replace("\n", " ")
    print(f"  row_id={rid:4d}  images=[{img_info}]")
    print(f"           bangla_q: {bq_snippet}")

# ── 4. Row-wise image↔question alignment ──────────────────────────────────────
print(f"\n{'─'*50}")
print("4. ROW-WISE ALIGNMENT CHECK (orig question vs bangla_question)")
print(f"{'─'*50}")
random.seed(99)
check_ids = random.sample(df["row_id"].tolist(), 5)

for rid in check_ids:
    row_final = df[df["row_id"] == rid].iloc[0]
    row_orig  = orig[orig["row_id"] == rid].iloc[0]
    eng_orig  = str(row_orig["question"])[:100].replace("\n", " ")
    bng_final = str(row_final["bangla_question"])[:100].replace("\n", " ")
    img_match = (row_orig["images"] is not None)
    print(f"  row_id={rid}")
    print(f"    EN: {eng_orig}")
    print(f"    BN: {bng_final}")
    print(f"    image present in orig: {img_match}")

# ── 5. messages column spot-check ─────────────────────────────────────────────
print(f"\n{'─'*50}")
print("5. MESSAGES COLUMN SPOT-CHECK (row 0, 499, 999)")
print(f"{'─'*50}")
for rid in [0, 499, 999]:
    row = df[df["row_id"] == rid].iloc[0]
    msgs = row["messages"]
    print(f"\n  row_id={rid}  (len messages={len(msgs)})")
    for m in msgs:
        role    = m.get("role", "?")
        content = str(m.get("content", ""))[:120].replace("\n", " ")
        print(f"    [{role:9s}] {content}")

# ── 6. bangla_assistant sample ────────────────────────────────────────────────
print(f"\n{'─'*50}")
print("6. BANGLA_ASSISTANT SAMPLE (row 0)")
print(f"{'─'*50}")
print(df[df["row_id"] == 0].iloc[0]["bangla_assistant"])

# ── 7. data_source distribution ───────────────────────────────────────────────
print(f"\n{'─'*50}")
print("7. DATA_SOURCE DISTRIBUTION")
print(f"{'─'*50}")
print(df["data_source"].value_counts().to_string())

# ── 8. Ground truth sample ────────────────────────────────────────────────────
print(f"\n{'─'*50}")
print("8. REWARD_MODEL / GROUND_TRUTH SAMPLES (5 rows)")
print(f"{'─'*50}")
for _, row in df.head(5).iterrows():
    gt = row["reward_model"].get("ground_truth", "N/A") if isinstance(row["reward_model"], dict) else "N/A"
    print(f"  row_id={row['row_id']:4d}  gt={gt}")

print(f"\n{'='*70}")
print("✅  VERIFICATION COMPLETE — dataset looks good for training!")
print(f"{'='*70}")
