import pandas as pd
import json

# Load the math parquet file
print("Loading dataset...")
df = pd.read_parquet("./dataset/DeepVision-103K/math-77k.parquet")
print(f"Total samples: {len(df)}")

# Parse domain-anno JSON and filter for geometry
def is_geometry(domain_anno_str):
    try:
        domain = json.loads(domain_anno_str)
        # Check if Level_1 is Geometry
        if domain.get("Level_1") == "Geometry":
            return True
        # Also check if Knowledge contains geometry-related terms
        knowledge = domain.get("Knowledge", [])
        geometry_keywords = ["Triangle", "Circle", "Polygon", "Geometry", "Angle", "Plane"]
        return any(any(keyword in k for keyword in geometry_keywords) for k in knowledge)
    except:
        return False

# Filter geometry samples
print("Filtering geometry samples...")
geometry_df = df[df["domain-anno"].apply(is_geometry)]
print(f"Filtered {len(geometry_df)} geometry samples")

# Save to parquet
output_file = "DeepVision_Geometry_10K.parquet"
geometry_df.to_parquet(output_file)
print(f"Saved to {output_file}")
print(f"\nSample questions:")
for i, q in enumerate(geometry_df["question"].head(3)):
    print(f"{i+1}. {q[:100]}...")