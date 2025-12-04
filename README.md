# Artificial Intelligence Project SKKU
2025-2 Team-4

## Quick Start

This section will guide you through setting up and running the recommendation system. This project implements a Hybrid Recommendation System combining **Content-Based Filtering**, **Collaborative Filtering (SVD)**, and **Graph Neural Networks (LightGCN)**, optimized via an MLP blender.

### 1. Installation

**Clone the Repository**

```bash
git clone https://github.com/sjjeon0925/skku-AI-project-25-2-team-4.git
cd skku-AI-project-25-2-team-4
```

**Set up Python Virtual Environment and Install Dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Setting Data

**Make `data` Folder and Move the following `4 csv files` into it:**

```text
./data/
  ├── menu_data.csv
  ├── rest_data.csv
  ├── user_data.csv
  └── rating_data.csv
```

**The `Data Structure` for Each File is as follows:**

1. **`menu_data.csv`**

| menu_id | menu | rest_id | price | features |
|-|-|-|-|-|
|1|삼겹|1|16000|육류 고기요리 한식 고기류 담백한 풍미 16000원대|
| num | str | num | num | str (words separated by space) |

2. **`rest_data.csv`**

| rest_id | rest_name | Latitude | Longitude | rating |
|-|-|-|-|-|
| 1 | ~고깃집 | 37.xx | 126.xx | 4.18 |
| num | str | num | num | num |

3. **`user_data.csv`**

| user_id | preference | allergy |
|-|-|-|
| 200 | 순대 순댓국 기타 부드러운 맛 | 새우 |
| num | str (words separated by space) | str (optional) |

4. **`rating_data.csv`**

| user_id | menu_id | rating | location | append |
|-|-|-|-|-|
| 201 | 84 | 2.95 | 후문 | 김밥 기타 바삭한 식감 5000원대 |
| num | num | num | str (location at time of rating) | str (additional query at time of rating) |

---

### 3. Train the Model

You can train the model using the `auto_optimizer.py` script. This script handles hyperparameter tuning and the final training process automatically.

#### Option A: Quick Run (Recommended)
Uses pre-defined optimal parameters to train the models immediately.

```bash
python auto_optimizer.py --task run
```

#### Option B: Hyperparameter Optimization
Performs Grid Search with Cross-Validation to find the best parameters before training.

```bash
python auto_optimizer.py --task opt
```

**Available Modes in Optimization:**
The system evaluates three different approaches:
1.  **Baseline**: Content-Based + Collaborative Filtering (SVD) + Meta Data -> MLP
2.  **GNN Only**: Graph Neural Network (LightGCN) only.
3.  **Proposed (Ensemble)**: CB + CF + **GNN** + Meta Data -> MLP (Best Performance).

**Model Architecture & Features (Input X for MLP)**

The **Proposed Model** uses a Supervised Learning approach (MLP) to ensemble various scores:

* **CF Score:** Predicted rating from Matrix Factorization (`SVD`).
* **CB Score:** Cosine Similarity between user preference and menu features (`TF-IDF`).
* **Graph Score:** User-Item affinity score learned by **LightGCN** (captures high-order connectivity).
* **Meta Features:**
    * **Distance:** Calculated score based on User's current location vs. Restaurant location.
    * **Price:** Normalized menu price.
    * **Restaurant Rating:** Average rating of the restaurant.

The trained models and scalers are saved in the `model/` directory.

---

### 4. Evaluate the Model

To evaluate the performance (Recall@10, Precision@10, NDCG@10) of the trained models:

```bash
python evaluate.py --mode proposed --model_name best_proposed
```
* `--mode`: Choose between `baseline`, `gnn_only`, or `proposed`.

---

### 5. User Menu Recommendations

The trained model is used to predict ratings for unrated menus and return the Top-N recommendations.

**Important:** You must use the correct `--mode` and `--model_name` corresponding to your training step.

#### Usage Syntax
```bash
python3 predict.py --i [USER_ID] --l [LOCATION] --mode [MODE] --model_name [NAME]
```

#### Examples by Mode

**1. Proposed Mode (Default & Best)**
Requires `best_proposed_mlp.keras` and `best_proposed_gnn.pth`.
```bash
python3 predict.py --i 2020123456 --l b --b 25000 --mode proposed --model_name best_proposed
```

**2. Baseline Mode**
Requires `best_baseline_mlp.keras`.
```bash
python3 predict.py --i 2020123456 --l b --b 25000 --mode baseline --model_name best_baseline
```

**3. GNN Only Mode**
Requires `best_gnn_only_gnn.pth`.
```bash
python3 predict.py --i 2020123456 --l b --b 25000 --mode gnn_only --model_name best_gnn_only
```

#### Arguments Detail

| Argument | Description | Usage in Code |
| - | - | - |
| `--i` | **Required:** Target User ID. | Used for **CF/GNN Score prediction** and looking up preferences. |
| `--l` | **Required:** Location code (`s`:Station, `b`:Back gate, `n`:North, `f`:Front). | Used to calculate **Distance Score** (Real-time accessibility). |
| `--mode` | **Required:** Model Type (`proposed`, `baseline`, `gnn_only`). | Determines feature dimensions and logic. |
| `--model_name` | **Required:** Prefix of the saved model files. | Loads specific model files from `model/` dir. |
| `--b` | **Optional:** Max budget (default: 100000). | Used for **Hard Filtering** (removes expensive menus). |

**Output Example**

The script outputs the Top 10 recommended menus with their predicted scores:

```text
[Top 10 Recommendations]
Menu                           | Restaurant           | Price      | Score
------------------------------------------------------------
김치찌개                       | XX식당               | 9,000      | 4.85
제육볶음                       | YY분식               | 8,000      | 4.82
...
```