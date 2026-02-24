# DeepBanglaMath — Sprint 1 Work Plan

**Date:** 24 February 2026

---

## Team

| Name | Chunk |
|------|-------|
| Khan Raiyan Ibne Reza | chunk_1_of_3.xlsx |
| [Name 1] | chunk_2_of_3.xlsx |
| [Name 2] | chunk_3_of_3.xlsx |

---

## Today (24 Feb) — before proposal submission

### Khan Raiyan Ibne Reza
- [ ] Locate `DeepVision_Geometry_Pilot_1K.parquet` in the project folder.
- [ ] Run `prepare_for_group.py` to split it into 3 Excel chunks.
- [ ] Take `chunk_1_of_3.xlsx` and translate the `bangla_question` column (minimum 300 rows).
- [ ] Share the other two chunks via Google Drive with translation instructions.
- [ ] After receiving translated chunks back, run the merge script → `DeepBanglaMath_Pilot_READY.parquet`.
- [ ] Open Unsloth Qwen3-VL-2B Colab notebook, load the READY file, run 1 epoch on 200 samples, screenshot the loss curve + one Bangla output.
- [ ] Update the project proposal with the READY file name and screenshot.

### [Name 1]
- [ ] Receive `chunk_2_of_3.xlsx`.
- [ ] Translate the `bangla_question` column (NCTB Bangla style).
- [ ] Finish at least 200–300 rows today, send the file back.

### [Name 2]
- [ ] Receive `chunk_3_of_3.xlsx`.
- [ ] Translate the `bangla_question` column (NCTB Bangla style).
- [ ] Finish at least 200–300 rows today, send the file back.

---

## 25–27 Feb

### Khan Raiyan Ibne Reza
- [ ] Merge any new chunks received → update `DeepBanglaMath_10K_READY.parquet`.
- [ ] Start fine-tuning on 4,000–5,000 rows (Colab, 2–3 sessions per day).
- [ ] Save checkpoints to Google Drive after every session.
- [ ] Extract 20–30 NCTB-style test questions for evaluation.
- [ ] Prepare 1-page progress slide for the course presentation.

### [Name 1] & [Name 2]
- [ ] Complete remaining rows in your chunk.
- [ ] Send final Excel file by 26 Feb evening.

---

## Before Final Submission

### Khan Raiyan Ibne Reza
- [ ] Run final evaluation (exact match, CoT score).
- [ ] Upload final dataset to Hugging Face (private).
- [ ] Prepare full report + code + dataset link.
- [ ] Submit everything.

---

Print this list and tick off items as you finish them.