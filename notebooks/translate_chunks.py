"""
translate_chunks.py
───────────────────────────────────────────────────────────────────────────────
Translates english_question → bangla_question (NCTB-style) using Gemini 2.0 Flash.

• Sends questions in BATCHES of 10 → ~10× faster than row-by-row
• Resume-safe: skips rows that already have a bangla_question
• Saves progress after every batch (safe to Ctrl+C and restart)
• Keeps LaTeX, <image>, geometry labels, MCQ structure intact

Usage
─────
  1. Set your API key:   set GEMINI_API_KEY=AIza...
     OR paste it directly into API_KEY below.
  2. Run:  .venv\Scripts\python.exe notebooks/translate_chunks.py
  3. To translate chunk_02 or 03, change CHUNK_FILE below.
"""

import os
import re
import time
import pandas as pd
import google.generativeai as genai

# ── CONFIG ────────────────────────────────────────────────────────────────────
API_KEY    = os.getenv("GEMINI_API_KEY", "YOUR_GEMINI_API_KEY_HERE")
MODEL      = "gemini-2.0-flash"
BATCH_SIZE = 10      # questions per API call  (max ~20 before response gets messy)
DELAY      = 1.0     # seconds between batches (free tier: 15 req/min → 4 s safe)

DATASET_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset", "10K Geometry", "chunks"
)

# ── Change this to chunk_02 or chunk_03 as needed ─────────────────────────────
CHUNK_FILE = os.path.join(DATASET_DIR, "chunk_01_rows_0-199.csv")

# ── NCTB SYSTEM PROMPT ────────────────────────────────────────────────────────
SYSTEM = """\
You are an expert Bangla math translator for NCTB (জাতীয় শিক্ষাক্রম ও পাঠ্যপুস্তক বোর্ড) textbooks.

STRICT RULES — follow every rule without exception:

1. Translate ALL English prose to formal/literary Bangla (no colloquial words).
2. Convert ALL Arabic/Western numerals to Bangla numerals:
   0→০  1→১  2→২  3→৩  4→৪  5→৫  6→৬  7→৭  8→৮  9→৯
   (including numbers inside sentences, e.g. "20 cm" → "২০ সেন্টিমিটার")
3. Keep LaTeX math EXACTLY unchanged: \\(x^2\\), $\\frac{a}{b}$, \\[...\\], etc.
4. Keep <image> tag exactly — do NOT translate or remove it.
5. Keep geometric point labels in English: A, B, C, D, O, P, Q …
6. Keep unit symbols as Bangla words: cm→সেন্টিমিটার, m→মিটার, sq.cm→বর্গ সেন্টিমিটার
7. MCQ options: keep the option letter exactly (A. / B. / (A) / (B) / A、 etc.),
   translate only the option text.
8. Special symbols (①②③ etc.) keep unchanged.
9. Return ONLY the translated text. NO explanations, NO English, NO extra lines.

Key vocabulary:
  triangle→ত্রিভুজ  circle→বৃত্ত  angle→কোণ  area→ক্ষেত্রফল  perimeter→পরিসীমা
  right triangle→সমকোণী ত্রিভুজ  equilateral→সমবাহু  isosceles→সমদ্বিবাহু
  radius→ব্যাসার্ধ  diameter→ব্যাস  height→উচ্চতা  base→ভূমি  side→বাহু
  square→বর্গক্ষেত্র  rectangle→আয়তক্ষেত্র  parallelogram→সামান্তরিক
  trapezoid→ট্রাপিজিয়াম  line segment→রেখাংশ  midpoint→মধ্যবিন্দু
  perpendicular→লম্ব  parallel→সমান্তরাল  congruent→সর্বসম  similar→সদৃশ
  figure→চিত্র  shaded→ছায়াযুক্ত  what is→কত / কী  find→নির্ণয় কর
  given→দেওয়া আছে  length→দৈর্ঘ্য  width→প্রস্থ  cube→ঘনক
  three-view→ত্রিদৃশ্য  front view→সামনের দৃশ্য  left view→বামের দৃশ্য
  top view→উপরের দৃশ্য
"""

# ── INIT GEMINI ───────────────────────────────────────────────────────────────
if API_KEY == "YOUR_GEMINI_API_KEY_HERE":
    raise SystemExit(
        "\n❌  Set your Gemini API key:\n"
        "    Windows:  set GEMINI_API_KEY=AIza...\n"
        "    Or edit API_KEY directly in this script."
    )

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(MODEL, system_instruction=SYSTEM)


def translate_batch(questions: list[str]) -> list[str]:
    """Send up to BATCH_SIZE questions in one request; return translations list."""
    prompt_lines = []
    for i, q in enumerate(questions, 1):
        prompt_lines.append(f"Q{i}: {q}")

    prompt = (
        f"Translate the following {len(questions)} math question(s) to NCTB-style Bangla.\n"
        f"Return EXACTLY in this format — one line per answer, starting with the number and pipe:\n"
        + "\n".join(f"{i}| <your translation>" for i in range(1, len(questions) + 1))
        + "\n\n"
        + "\n\n".join(prompt_lines)
    )

    response = model.generate_content(prompt)
    raw = response.text.strip()

    # Parse "1| ..." lines
    results = {}
    for line in raw.splitlines():
        m = re.match(r"^(\d+)\s*\|\s*(.+)$", line.strip())
        if m:
            results[int(m.group(1))] = m.group(2).strip()

    # Fall back to sequential lines if parsing failed
    if len(results) < len(questions):
        lines = [l.strip() for l in raw.splitlines() if l.strip()]
        results = {i + 1: lines[i] for i in range(min(len(lines), len(questions)))}

    return [results.get(i + 1, "") for i in range(len(questions))]


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    if not os.path.exists(CHUNK_FILE):
        raise FileNotFoundError(f"Chunk file not found: {CHUNK_FILE}")

    df = pd.read_csv(CHUNK_FILE, dtype=str).fillna("")
    total   = len(df)
    pending = df[df["bangla_question"].str.strip() == ""].index.tolist()

    print(f"Chunk   : {os.path.basename(CHUNK_FILE)}")
    print(f"Total   : {total} rows")
    print(f"Done    : {total - len(pending)} rows  (will skip)")
    print(f"Pending : {len(pending)} rows  (will translate now)")
    print(f"Model   : {MODEL}   Batch size: {BATCH_SIZE}\n")

    if not pending:
        print("✅ All rows already translated. Nothing to do.")
        return

    processed = 0
    errors    = 0

    for chunk_start in range(0, len(pending), BATCH_SIZE):
        batch_idx = pending[chunk_start : chunk_start + BATCH_SIZE]
        questions = [str(df.at[i, "english_question"]) for i in batch_idx]

        print(f"[{processed + 1}–{processed + len(batch_idx)}/{len(pending)}]  translating ...", end=" ", flush=True)

        try:
            translations = translate_batch(questions)
            for i, bangla in zip(batch_idx, translations):
                df.at[i, "bangla_question"] = bangla
            processed += len(batch_idx)
            print(f"✅  (batch {chunk_start // BATCH_SIZE + 1})")
        except Exception as e:
            errors += 1
            print(f"❌  Error: {e}")
            time.sleep(5)   # back off on error

        # Save after every batch — safe to interrupt & resume
        df.to_csv(CHUNK_FILE, index=False, encoding="utf-8-sig")

        if chunk_start + BATCH_SIZE < len(pending):
            time.sleep(DELAY)

    print(f"\n{'─'*60}")
    print(f"✅  Translated : {processed}")
    print(f"❌  Errors     : {errors}")
    print(f"💾  Saved to   : {CHUNK_FILE}")


if __name__ == "__main__":
    main()
