# 인공지능 프로젝트 skku
2025-2 team-4

## Quick Start

이 섹션은 추천 시스템을 설정하고 실행하는 방법을 안내합니다. 본 프로젝트는 **콘텐츠 기반 필터링**, **협업 필터링 (SVD)**, 그리고 **그래프 신경망 (LightGCN)**을 결합하고 MLP로 최적화한 **하이브리드 추천 시스템**을 구현합니다.

### 1. Installation

**저장소 복제 (Clone the Repository)**

```bash
git clone https://github.com/sjjeon0925/skku-AI-project-25-2-team-4.git
cd skku-AI-project-25-2-team-4
```

**Python 가상 환경 설정 및 종속성 설치**

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Setting Data

**`data` 폴더를 생성하고 아래 4개의 CSV 파일을 해당 폴더로 이동**

```text
./data/
  ├── menu_data.csv
  ├── rest_data.csv
  ├── user_data.csv
  └── rating_data.csv
```

**각 파일의 데이터 구성은 다음과 같습니다.**

1. **`menu_data.csv`**

| menu_id | menu | rest_id | price | features |
| - | - | - | - | - |
|1|삼겹|1|16000|육류 고기요리 한식 고기류 담백한 풍미 16000원대|
| num | str | num | num | str (단어는 공백으로 구분) |

2. **`rest_data.csv`**

| rest_id | rest_name | Latitude | Longitude | rating |
| - | - | - | - | - |
| 1 | ~고깃집 | 37.xx | 126.xx | 4.18 |
| num | str | num | num | num |

3. **`user_data.csv`**

| user_id | preference | allergy |
| - | - | - |
| 200 | 순대 순댓국 기타 부드러운 맛 | 새우 |
| num | str (단어는 공백으로 구분) | str (선택 사항) |

4. **`rating_data.csv`**

| user_id | menu_id | rating | location | append |
| - | - | - | - | - |
| 201 | 84 | 2.95 | 후문 | 김밥 기타 바삭한 식감 5000원대 |
| num | num | num | str (평가 당시 위치) | str (평가 당시 추가 쿼리) |

---

### 3. Train the Model

모델 학습은 `auto_optimizer.py` 스크립트를 통해 수행됩니다. 이 스크립트는 하이퍼파라미터 튜닝과 최종 학습 과정을 자동으로 처리합니다.

#### 옵션 A: 빠른 실행 (Quick Run - 권장)
사전에 정의된 최적 파라미터를 사용하여 즉시 모델을 학습합니다.

```bash
python auto_optimizer.py --task run
```

#### 옵션 B: 하이퍼파라미터 최적화 (Optimization)
Grid Search와 교차 검증(Cross-Validation)을 수행하여 최적의 파라미터를 찾은 뒤 학습합니다.

```bash
python auto_optimizer.py --task opt
```

**최적화 모드별 모델 설명:**
시스템은 다음 세 가지 접근 방식을 평가하고 학습합니다.
1.  **Baseline**: 콘텐츠 기반(CB) + 협업 필터링(SVD) + 메타 데이터 -> MLP
2.  **GNN Only**: 그래프 신경망 (LightGCN) 단독 모델.
3.  **Proposed (Ensemble)**: CB + CF + **GNN** + 메타 데이터 -> MLP (**최고 성능**)

**제안 모델(Proposed) 구조 및 입력 특징 (Input X)**

제안된 모델은 **지도 학습 (MLP)** 방식을 사용하여 다양한 추천 점수를 앙상블합니다.

* **CF 점수:** 행렬 분해(`SVD`) 모델이 예측한 사용자-아이템 예상 평점.
* **CB 점수:** 사용자 선호도와 메뉴 특징 간의 `코사인 유사도` 점수 (`TF-IDF`).
* **Graph 점수:** **LightGCN**으로 학습된 사용자-아이템 간의 친밀도 점수 (고차 연결성 포착).
* **메타 특징:**
    * **거리:** 사용자의 현재 위치와 식당 위치 간의 거리 점수.
    * **가격:** 정규화된 메뉴 가격.
    * **식당 평점:** 식당의 평균 평점.

학습된 모델과 스케일러 파일은 `model/` 폴더에 저장됩니다.

---

### 4. Evaluate the Model

`evaluate.py`를 통해 학습된 모델의 성능 지표(Recall@10, Precision@10, NDCG@10)를 평가할 수 있습니다.

```bash
python evaluate.py --mode proposed --model_name best_proposed
```
* `--mode`: `baseline`, `gnn_only`, `proposed` 중 선택.

---

### 5. User Menu Recommendations

학습된 모델을 사용하여 사용자가 아직 평가하지 않은 메뉴들의 평점을 예측하고, 상위 N개를 추천합니다.

**중요:** 학습 단계에서 사용한 `--mode`와 저장된 `--model_name`을 정확히 일치시켜야 합니다.

#### 사용 구문
```bash
python3 predict.py --i [USER_ID] --l [LOCATION] --mode [MODE] --model_name [NAME]
```

#### 모드별 실행 예시

**1. Proposed 모드 (기본값 & 권장)**
`best_proposed_mlp.keras` 와 `best_proposed_gnn.pth` 파일이 필요합니다.
```bash
python3 predict.py --i 2020311640 --l b --b 25000 --mode proposed --model_name best_proposed
```

**2. Baseline 모드**
`best_baseline_mlp.keras` 파일이 필요합니다.
```bash
python3 predict.py --i 2020311640 --l b --b 25000 --mode baseline --model_name best_baseline
```

**3. GNN Only 모드**
`best_gnn_only_gnn.pth` 파일이 필요합니다.
```bash
python3 predict.py --i 2020311640 --l b --b 25000 --mode gnn_only --model_name best_gnn_only
```

#### 인자(Arguments) 설명

| Argument | 설명 | 코드 내 사용 방식 |
| - | - | - |
| `--i` | **필수:** 추천을 요청하는 유저 ID. | **CF/GNN 점수 예측** 및 사용자 선호도 조회에 사용됩니다. |
| `--l` | **필수:** 현재 위치 코드 (`s`:성균관대역, `b`:후문, `n`:북문, `f`:정문). | **거리 점수 계산** (실시간 접근성)에 사용됩니다. |
| `--mode` | **필수:** 모델 타입 (`proposed`, `baseline`, `gnn_only`). | 입력 특징(Feature)의 차원 수와 로직을 결정합니다. |
| `--model_name` | **필수:** 저장된 모델 파일의 접두사(prefix). | `model/` 폴더에서 특정 모델 파일을 로드하는 데 사용됩니다. |
| `--b` | **선택:** 최대 예산 (기본값: 100000). | **하드 필터링**을 통해 예산을 초과하는 메뉴를 제외합니다. |

**출력 예시**

스크립트는 필터링을 통과한 메뉴 중 예상 평점이 가장 높은 Top 10을 출력합니다.

```text
[Top 10 Recommendations]
Menu                           | Restaurant           | Price      | Score
------------------------------------------------------------
김치찌개                       | XX식당               | 9,000      | 4.85
제육볶음                       | YY분식               | 8,000      | 4.82
...
```