import pandas as pd
import json
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Custom JSON encoder to handle numpy arrays and bytes
class SafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bytes):
            return f"<bytes len={len(obj)}>"
        return super().default(obj)

# ====================== LOAD YOUR FILE ======================
# Change filename if you want to inspect Pilot_1K instead
df = pd.read_parquet("DeepVision_Geometry_10K.parquet")

print("=== BASIC INFO ===")
print("Shape (rows, columns):", df.shape)
print("Columns:", df.columns.tolist())

# ====================== FIRST ROW EXAMPLE ======================
row = df.iloc[0]          # change 0 to any number 0–9999

print("\n=== ROW 0 FULL STRUCTURE ===")
print(row)

print("\n=== PROMPT (what the model sees) ===")
print(json.dumps(row["prompt"], indent=2, cls=SafeEncoder))   # pretty print the list of dicts

print("\n=== QUESTION TEXT (extracted) ===")
print(row["question"][:500] + "..." if len(row["question"]) > 500 else row["question"])

print("\n=== REWARD_MODEL (correct answer) ===")
print(row["reward_model"])

print("\n=== ANNOTATION EXAMPLE (q32-vision-anno) ===")
anno = json.loads(row["q32-vision-anno"])   # convert JSON string → Python dict
print(json.dumps(anno, indent=2)[:800] + "...")  # first part only

# ====================== SHOW THE ACTUAL IMAGE ======================
img_entry = row["images"][0]    # dict with 'bytes' and 'path' keys
img_bytes = img_entry["bytes"] if isinstance(img_entry, dict) else img_entry
img = Image.open(io.BytesIO(img_bytes))
print("\nImage size:", img.size, "Format:", img.format)

# Save and show
img.save("sample_geometry_image.png")
img.show()                      # opens in your default image viewer
plt.imshow(img)
plt.title("Sample Diagram from Row 0")
plt.axis("off")
plt.show()