# DeepVision Geometry Dataset - Data Structure Documentation

## 📊 Dataset Overview

This dataset contains **1,000 pilot geometry samples** extracted from the DeepVision-103K dataset, specifically filtered for geometry-related mathematical problems with visual elements.

### Files in this Project

| File | Rows | Description |
|------|------|-------------|
| `DeepVision_Geometry_Pilot_1K.parquet` | 1,000 | Pilot dataset for demonstrations and proposals |
| `DeepVision_Geometry_10K.parquet` | 58,735 | Full geometry dataset filtered from math-77k |
| `dataset/DeepVision-103K/math-77k.parquet` | 77,135 | Original math dataset (source) |
| `dataset/DeepVision-103K/visual_logic-26k.parquet` | ~26,000 | Original visual logic dataset (not used) |

---

## 📋 Column Structure

The dataset contains **10 columns** organized into the following categories:

### 1. Core Problem Data

#### `question` (string)
- **Description**: The raw text of the mathematical problem
- **Format**: Plain text with possible LaTeX notation (e.g., `\(x^2\)`, `$\box{...}$`)
- **Example**: `"In the figure, triangle ABC is a right triangle. The area of the shaded part ① is 28 square centimeters smaller than the area of the shaded part ②. Given that AB = 20 centimeters, what is the length of BC?"`
- **Usage**: Primary input text for the model

#### `images` (list of structs)
- **Description**: Image data associated with the problem
- **Structure**: 
  ```python
  [
    {'bytes': binary_data, 'path': 'image_path.png'}
  ]
  ```
- **Format**: Each element contains:
  - `bytes`: Binary PNG/JPEG image data
  - `path`: Original image filename/path
- **Usage**: Visual input for multimodal models
- **Note**: Most samples contain 1 image; some may have multiple

---

### 2. Problem Classification & Metadata

#### `data_source` (string)
- **Description**: Origin dataset identifier
- **Value**: `"math-77k"` (constant for this filtered dataset)
- **Usage**: Tracking data provenance

#### `ability` (string)
- **Description**: Core skill category
- **Value**: `"math"` (constant for this filtered dataset)
- **Usage**: High-level task classification

#### `domain-anno` (JSON string)
- **Description**: Hierarchical topic classification with knowledge concepts
- **Structure**:
  ```json
  {
    "Knowledge": ["Area of Triangle", "Area of Circle", "Four Operations with Decimals"],
    "Level_1": "Geometry",
    "Level_2": "Plane Geometry",
    "Level_3": "Basic Plane Figures",
    "Level_4": "Triangle"
  }
  ```
- **Fields**:
  - `Knowledge`: List of specific mathematical concepts/skills required
  - `Level_1`: Broad domain (e.g., Geometry, Algebra, Probability and Statistics)
  - `Level_2`: Sub-domain (e.g., Plane Geometry, Solid Geometry)
  - `Level_3`: Topic category (e.g., Basic Plane Figures, Triangle)
  - `Level_4`: Specific topic (e.g., Sine and Cosine Laws, Area Calculations)
- **Usage**: Curriculum alignment, difficulty estimation, topic-based filtering

---

### 3. Model Training & Evaluation

#### `prompt` (list of structs)
- **Description**: Conversation format for training LMMs (Large Multimodal Models)
- **Structure**:
  ```python
  [
    {'content': '...system instructions...', 'role': 'system'},
    {'content': '<image>...question text...', 'role': 'user'}
  ]
  ```
- **Format**: OpenAI-style chat format with roles:
  - `system`: Instructions for the model's behavior
  - `user`: The actual problem (includes `<image>` placeholder + question text)
- **Usage**: Direct input for instruction-tuned models (GPT, Qwen, etc.)
- **Note**: The `<image>` token indicates where visual input should be processed

#### `reward_model` (struct)
- **Description**: Ground truth and evaluation data for RL/RLHF training
- **Structure**:
  ```python
  {
    'ground_truth': '18.5',
    'equivalent_answers': ['37/2', '18.5'],
    'style': 'rule'
  }
  ```
- **Fields**:
  - `ground_truth`: Canonical correct answer
  - `equivalent_answers`: List of acceptable answer formats (fractions, decimals, etc.)
  - `style`: Answer validation method (`'rule'` = rule-based matching)
- **Usage**: 
  - Supervised fine-tuning (SFT) labels
  - Reward model training for RLHF
  - Answer verification during evaluation

#### `mimo-pass_rate` (string)
- **Description**: Model performance metric from MIMO evaluation
- **Format**: Fraction string (e.g., `"2/8"` means 2 out of 8 models solved correctly)
- **Usage**: Difficulty estimation and sample selection
- **Interpretation**: Lower pass rate = harder problem

---

### 4. Rich Annotations for Analysis

#### `gpt5-mini-vision-anno` (JSON string)
- **Description**: GPT-5-mini's visual element annotation
- **Structure**:
  ```json
  {
    "visual_elements": {
      "planar_geometry": ["Right Triangle", "Semicircle", "Chord"],
      "solid_geometry": [...],
      "analytic_plots": [...],
      "data_charts": [...],
      "schematic_diagrams": [...],
      "real_world_items": [...]
    }
  }
  ```
- **Usage**: 
  - Visual diversity analysis
  - Filtering by visual element type
  - Understanding image content without loading images

#### `q32-vision-anno` (JSON string)
- **Description**: Qwen3.2-vision's detailed visual complexity analysis
- **Structure**:
  ```json
  {
    "image_necessity": "required",
    "visual_complexity": {
      "annotation_density_analysis": "Moderate annotation density...",
      "element_richness_analysis": "Contains a right triangle ABC...",
      ...
    }
  }
  ```
- **Fields**:
  - `image_necessity`: Whether image is required to solve (`"required"` / `"optional"`)
  - `visual_complexity`: Detailed analysis of visual elements, annotations, and complexity
- **Usage**: 
  - Filtering image-dependent problems
  - Complexity-based curriculum learning
  - Understanding visual reasoning requirements

---

## 🔗 Column Relationships & Data Flow

### Training Pipeline Connections

```
┌─────────────────────────────────────────────────────────────┐
│                    MULTIMODAL MODEL TRAINING                 │
└─────────────────────────────────────────────────────────────┘

INPUT STAGE:
┌──────────────┐     ┌─────────────┐     ┌──────────────────┐
│   images     │────▶│   prompt    │────▶│  Model Input     │
│  (visual)    │     │  (text)     │     │  (multimodal)    │
└──────────────┘     └─────────────┘     └──────────────────┘
                           │
                           │ combines with
                           ▼
                    ┌─────────────┐
                    │  question   │
                    └─────────────┘

SUPERVISION STAGE:
┌──────────────────┐     ┌─────────────────────────────────┐
│  reward_model    │────▶│  Ground Truth Matching          │
│  .ground_truth   │     │  (for loss calculation)         │
│  .equivalent_    │     │                                 │
│   answers        │     │  Verification                   │
└──────────────────┘     └─────────────────────────────────┘

FILTERING/CURRICULUM STAGE:
┌──────────────────┐     ┌─────────────────────────────────┐
│  domain-anno     │────▶│  Topic-based Batching           │
│  mimo-pass_rate  │     │  Difficulty-based Ordering      │
│  q32-vision-anno │     │  Visual Complexity Filtering    │
└──────────────────┘     └─────────────────────────────────┘
```

### Key Relationships

1. **Question ↔ Images ↔ Prompt**
   - `question` is the raw problem text
   - `images` contains the visual diagram
   - `prompt` combines both in chat format with `<image>` token + question text

2. **Domain-anno ↔ Mimo-pass_rate**
   - `domain-anno.Level_4` correlates with `mimo-pass_rate`
   - More specific topics (Level_4) often have varied difficulty
   - Use together for curriculum learning (easy → hard)

3. **GPT5-mini-vision-anno ↔ Q32-vision-anno**
   - Both annotate visual content but serve different purposes:
     - GPT5: **What** visual elements exist (categorization)
     - Q32: **How complex** and **how necessary** the visual is (analysis)
   - Combine for comprehensive visual understanding

4. **Reward_model ↔ All fields**
   - Ground truth for evaluating model outputs
   - `equivalent_answers` handles multiple valid formats
   - Critical for both SFT and RLHF training

---

## 🎯 Model Training Usage

### 1. **Supervised Fine-Tuning (SFT)**

```python
import pandas as pd
from PIL import Image
import io

df = pd.read_parquet('DeepVision_Geometry_Pilot_1K.parquet')

for idx, row in df.iterrows():
    # Load image
    image_data = row['images'][0]['bytes']
    image = Image.open(io.BytesIO(image_data))
    
    # Get conversation prompt
    messages = row['prompt']  # Already in chat format
    
    # Get ground truth
    ground_truth = row['reward_model']['ground_truth']
    
    # Train model with (messages, image) → ground_truth
```

### 2. **Reinforcement Learning (RLHF/RLVR)**

```python
# Use reward_model for verification
def verify_answer(model_output, sample):
    ground_truth = sample['reward_model']['ground_truth']
    equivalent = sample['reward_model']['equivalent_answers']
    
    # Check if output matches any acceptable answer
    return model_output in equivalent or model_output == ground_truth
```

### 3. **Curriculum Learning**

```python
import json

# Sort by difficulty (pass rate)
def get_difficulty(pass_rate_str):
    num, denom = map(int, pass_rate_str.split('/'))
    return num / denom

df['difficulty'] = df['mimo-pass_rate'].apply(get_difficulty)
df_sorted = df.sort_values('difficulty', ascending=False)  # Easy → Hard

# Or filter by topic
df['level_1'] = df['domain-anno'].apply(lambda x: json.loads(x)['Level_1'])
geometry_only = df[df['level_1'] == 'Geometry']
```

### 4. **Visual Complexity Filtering**

```python
import json

# Filter for image-required problems
def is_image_required(q32_anno):
    anno = json.loads(q32_anno)
    return anno.get('image_necessity') == 'required'

image_required_df = df[df['q32-vision-anno'].apply(is_image_required)]
```

---

## 📊 Statistics Summary

| Metric | Value |
|--------|-------|
| Total Samples | 1,000 |
| Domain | 100% Geometry |
| Avg Images per Sample | ~1 |
| Answer Format | Mixed (decimals, fractions) |
| Visual Element Coverage | Planar geometry, solid geometry, diagrams |

---

## 🚀 Quick Start

```python
import pandas as pd
import json
from PIL import Image
import io

# Load dataset
df = pd.read_parquet('DeepVision_Geometry_Pilot_1K.parquet')

# Explore first sample
sample = df.iloc[0]

# View question
print("Question:", sample['question'])

# View image
image = Image.open(io.BytesIO(sample['images'][0]['bytes']))
image.show()

# View topic classification
domain = json.loads(sample['domain-anno'])
print("Topic:", domain['Level_1'], "→", domain['Level_4'])
print("Knowledge:", domain['Knowledge'])

# View answer
print("Ground Truth:", sample['reward_model']['ground_truth'])
print("Acceptable Answers:", sample['reward_model']['equivalent_answers'])
```

---

## 📝 Notes for Model Training

1. **Image Processing**: Images are stored as binary PNG/JPEG. Extract using `io.BytesIO()`
2. **Answer Format**: Use `equivalent_answers` to handle multiple valid formats
3. **Prompt Format**: Pre-formatted for chat-based models; modify system message as needed
4. **Visual Analysis**: Use GPT5/Q32 annotations for filtering without loading all images
5. **Difficulty Ordering**: Lower `mimo-pass_rate` = harder problem (use for curriculum)

---

## 📚 Citation

Original dataset: [DeepVision-103K](https://huggingface.co/datasets/skylenage/DeepVision-103K)

Paper: DeepVision-103K: A Visually Diverse, Broad-Coverage, and Verifiable Mathematical Dataset for Multimodal Reasoning
