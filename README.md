# 인공지능 프로젝트 skku
2025-2 team-4

## Quick Start

This section will guide you through setting up and running the model. Follow these steps to get started.

### 1. Installation

**Clone the Repository**

```bash
git clone https://github.com/sjjeon0925/skku-AI-project-25-2-team-4.git
cd skku-AI-project-25-2-team-4
```

**Set up Python Virtual Environment and Install Dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Setting Data

**Make `data` Folder and Move below `4 csv files` to the Folder**

```bash
menu_data.csv
rest_data.csv
user_data.csv
rating_data.csv
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

**Rating Data Structure**

The `rating_data.csv` serves as the basis for **Collaborative Filtering (CF)** training, thus the data must maintain `sparsity`.
* CF Model Validity: To ensure the CF model can predict expected ratings for **unrated items** (`I_unrated`), the data should be structured so that the entire set of users has NOT rated every single menu item.
* Current Dataset Standard: (e.g., **20 users** should only rate **15-20** of the 100 available menu items each). If the data is `100% dense`, the core objective of the CF model (predicting missing values) is lost.

### 3. Train the Model

The model training is performed by running `train.py`

```bash
python train.py
```

**Dataset Composition and Usage**

`train.py` utilizes a **Supervised Learning** approach to train the final `MLP` model.
* Ground Truth Data (Y): The `rating` column from `rating_data.csv` (**actual rating** left by the user) is used as the target vector (`Y`) that the MLP model is trained to predict.
* Dataset Split: The feature matrix (`X`) and target vector (`Y`) based on the entire `rating_data.csv` are randomly split into **80% (Train)**, **20% (Validation/Test)** before training begins. The weights of the MLP model are optimized using this training data.
* Role of the 5 Scores (MLP Input Features X):
    * **CF Score:** The expected rating for the user-item pair, predicted by the `SVD` model learned from `rating_data.csv`.
    * **CB Score:** The `Cosine Similarity` score between the user's preference vector and the menu's features vector.
    * **Distance/Price/Restaurant:** The menu's `price`, the restaurant's `location` (Latitude, Longitude), and the restaurant's `average rating` are normalized and used as input features for the MLP.

The trained model data and records of the loss change during training are stored in the `model` folder.

### 4. User Menu Recommendations

The trained model is used to process the user's request query.

```bash
python3 predict.py --i=2020311640 --l=b --b=25 --q="한식 얼큰한 육류"
```

| Argument | Description | Usage in Code |
| - | - | - |
| `--i` (User ID) | **Required:** Predictable ID of the user requesting the recommendation. | Used as the subject for **CF Score prediction** and for looking up the user's existing preference in `user_data.csv`. |
| `--l` (Location) | **Required:** User's current location code (`s`:성균관대역, `b`:후문, `n`:북문, `f`:정문). | Used in the **Distance Score calculation** to derive a **real-time accessibility score** for candidate menus. |
| `--b` (Budget) | **Required:** User's maximum budget (in thousands of KRW). | Directly used in **Hard Filtering** to exclude menus exceeding the budget from the candidate list. |
| `--q` (Query) | **Optional:** Keywords reflecting the user's immediate craving. | Used in **CB Score calculation**, where it is **combined** with `user_data.preference` to form the final content preference vector. |

Output Content Explanation
`predict.py` outputs the Top N menus by inputting the **hard-filtered** candidate menus into the `MLP` model to calculate the **Expected Rating** (`Predicted_Rating`), then sorting the results.
* Output Items: Menu Name, Restaurant Name, Price, Predicted Rating.
* Unrated Filtering: The current recommendation logic only lists items that the user has **NOT rated previously**.
    * (The unrated filtering can be removed later to allow previously rated items to appear in the recommendation list.)