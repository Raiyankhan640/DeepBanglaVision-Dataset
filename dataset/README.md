# DeepBanglaVision Dataset Organization

This repository contains **Bengali visual geometry math datasets** for training multimodal reasoning models.

## 📁 Folder Structure

```
dataset/
├── 1K Geometry/          ✅ Fully translated (1,000 rows)
│   ├── DeepBanglaMath_1K_translated.parquet   (8.5 MB)  ← Final dataset
│   └── chunks/                                          ← 5 translated Excel files
│       ├── chunk_01_rows_0-199.csv.xlsx
│       ├── chunk_02_rows_200-399.csv.xlsx
│       ├── ... (3 more)
│
├── 5K Geometry/          ⏳ Ready for translation (5,000 rows)
│   └── chunks/                                          ← 25 CSV files to translate
│       ├── chunk_01_rows_0-199.csv
│       ├── chunk_02_rows_200-399.csv
│       ├── ... (23 more)
│
└── 10K Geometry/         🔄 Partial (10K total, 1K translated)
    ├── DeepBanglaMath_1000_translated.parquet (8.5 MB)  ← First 1000 translated
    └── chunks/
        ├── chunk_01_rows_0-199.csv.xlsx  (translated)
        ├── ... (4 more translated)
        └── chunk_06_rows_1000-1199.csv   (empty, ready to translate)
```

---

## 🗃️ What's Included in Git

| File Type | Size | Status |
|-----------|------|--------|
| **Translated parquets** | ~8 MB each | ✅ Pushed to GitHub |
| **Excel/CSV chunks** | 30-50 KB each | ✅ Pushed to GitHub |
| **Raw parquets** (with images) | 40-500 MB each | ❌ Excluded (too large) |
| **Ready-to-translate** parquets | 40-500 MB each | ❌ Excluded (too large) |

> **Note:** Large raw parquets (40-500 MB) are excluded via `.gitignore` because they exceed GitHub's 100MB file limit. These can be downloaded separately from [source](https://huggingface.co/datasets/wendlerc/DeepVision-103K) or regenerated using `organize_datasets.py`.

---

## 📊 Dataset Details

### **1K Geometry** (✅ Complete)
- **Total rows:** 1,000
- **Columns:** 14
  - Original: `question`, `images`, `prompt`, `reward_model`, ...
  - Bengali: `bangla_question`, `bangla_assistant`, `messages`
- **Format:** ChatML-ready for training
- **Use case:** Initial model training & validation

### **5K Geometry** (⏳ Ready for Translation)
- **Total rows:** 5,000
- **Status:** Raw English questions exported to 25 CSV chunks
- **Next step:** Translate the chunks → merge with `build_final_dataset.py`

### **10K Geometry** (🔄 Partial)
- **Total rows:** 10,000 (source has 58,735 geometry questions)
- **Translated:** First 1,000 rows
- **Status:** Can translate remaining 9,000 rows in batches

---

## 🛠️ Workflow

### **1. Extract Raw Data**
```bash
# Extract N rows from source → raw parquet
python notebooks/organize_datasets.py
```

### **2. Export Translation Chunks**
```bash
# Create CSV chunks with 3 columns: row_id, english_question, bangla_question
python notebooks/export_chunks.py
```

### **3. Translate** 
- **Manual:** Open Excel, translate `bangla_question` column
- **Automated:** Use `translate_chunks.py` (Gemini 2.0 Flash API)

### **4. Build Final Dataset**
```bash
# Merge translated chunks + original parquet → final ChatML dataset
python notebooks/build_final_dataset.py
```

### **5. Verify**
```bash
# Check data integrity, image alignment, null values
python notebooks/verify_dataset.py
```

---

## 📦 Final Dataset Column Schema

| Column | Type | Description |
|--------|------|-------------|
| `row_id` | int64 | Sequential index (0, 1, 2, ...) |
| `question` | str | Original English question |
| `bangla_question` | str | Translated Bengali question |
| `images` | object | PIL Image (embedded in parquet) |
| `prompt` | object | Array of dicts (system + user messages) |
| `reward_model` | object | `{"ground_truth": "18.5", "equivalent_answers": [...]}` |
| `bangla_assistant` | str | Templated Bengali answer with `\boxed{gt}` |
| `messages` | object | ChatML format: `[system, user, assistant]` |
| ... | ... | Additional metadata columns |

---

## 🚀 Training-Ready Format

Each row's `messages` column contains:

```python
[
  {
    "role": "system",
    "content": "You are a multimodal reasoning assistant..."
  },
  {
    "role": "user",
    "content": "<image>চিত্রে, ABC একটি সমকোণী ত্রিভুজ। ছায়াযুক্ত অংশ ① এর ক্ষেত্রফল..."
  },
  {
    "role": "assistant",
    "content": "চিত্রটি ভালোভাবে দেখে ধাপে ধাপে সমাধান করি।\n\n...\n\nসুতরাং, সঠিক উত্তর \\boxed{18.5}"
  }
]
```

Direct compatibility with: 
- LLaVA
- Qwen-VL
- InternVL
- Any ChatML-based vision-language model

---

## 📝 Scripts Reference

| Script | Purpose |
|--------|---------|
| `organize_datasets.py` | Create 1K/5K/10K folder structure |
| `export_chunks.py` | Export 3-column CSV chunks |
| `translate_chunks.py` | Batch translate via Gemini API |
| `build_final_dataset.py` | Merge translations → final parquet |
| `verify_dataset.py` | Comprehensive verification |

---

## 🔗 Source

**Original dataset:** [wendlerc/DeepVision-103K](https://huggingface.co/datasets/wendlerc/DeepVision-103K)  
**Geometry subset:** 58,735 questions from `math-77k.parquet`

---

## 📄 License

Same as source dataset (check [DeepVision-103K](https://huggingface.co/datasets/wendlerc/DeepVision-103K) for terms).
