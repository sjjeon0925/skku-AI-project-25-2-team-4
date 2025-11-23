# 인공지능 프로젝트 skku
2025-2 team-4

## Quick Start

이 섹션은 모델을 설정하고 실행하는 방법을 안내합니다. 시작하려면 다음 단계를 따르세요.

### 1. Installation

**저장소 복제 (Clone the Repository)**

```bash
git clone [https://github.com/sjjeon0925/skku-AI-project-25-2-team-4.git](https://github.com/sjjeon0925/skku-AI-project-25-2-team-4.git)
cd skku-AI-project-25-2-team-4
```

**Python 가상 환경 설정 및 종속성 설치**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Setting Data

**`data` 폴더를 생성하고 아래 4개의 CSV 파일을 해당 폴더로 이동**

```bash
menu_data.csv
rest_data.csv
user_data.csv
rating_data.csv
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

**평가 데이터 구조**

`rating_data.csv`는 **협업 필터링 (CF)** 학습의 기반이 되므로, 데이터는 `희소성 (sparsity)`을 유지해야 합니다.
* CF 모델 유효성: CF 모델이 **미평가 항목** (`I_unrated`)에 대한 예상 평점을 예측할 수 있도록 보장하기 위해, 전체 사용자가 모든 메뉴 항목에 평점을 매기지 않도록 데이터가 구성되어야 합니다.
* 현재 데이터셋 기준: (예: **20명의 사용자**는 100개의 사용 가능한 메뉴 항목 중 **각각 15~20개**만 평가해야 합니다). 데이터가 `100% 밀집`되어 있으면 CF 모델의 핵심 목표(결측값 예측)가 상실됩니다.

### 3. Train the Model

모델 학습은 `train.py`를 실행하면 수행됩니다.

```bash
python train.py
```

**데이터셋 구성 및 사용**

`train.py`는 최종 `MLP` 모델을 학습시키기 위해 **지도 학습 (Supervised Learning)** 접근 방식을 활용합니다.
* 정답 데이터 (Y): `rating_data.csv`의 `rating` 컬럼 (**실제 사용자 평점**)이 MLP 모델이 예측하도록 학습되는 타겟 벡터 (`Y`)로 사용됩니다.
* 데이터셋 분리: 전체 `rating_data.csv`를 기반으로 한 특징 행렬 (`X`)과 타겟 벡터 (`Y`)는 학습 시작 전에 **80% (훈련)**, **20% (검증/테스트)**로 무작위 분할됩니다. MLP 모델의 가중치는 이 훈련 데이터를 사용하여 최적화됩니다.
* 5가지 점수의 역할 (MLP 입력 특징 X):
    * **CF 점수:** `rating_data.csv`로부터 학습된 `SVD` 모델이 예측한 사용자-아이템 쌍에 대한 예상 평점.
    * **CB 점수:** 사용자의 선호도 벡터와 메뉴의 특징 벡터 간의 `코사인 유사도` 점수.
    * **거리/가격/식당:** 메뉴의 `가격`, 식당의 `위치` (위도, 경도), 식당의 `평균 평점`이 정규화되어 MLP의 입력 특징으로 사용됩니다.

학습된 모델 데이터와 학습 중 손실 변화 기록은 `model` 폴더에 저장됩니다.

### 4. User Menu Recommendations

학습된 모델은 사용자의 요청 쿼리를 처리하는 데 사용됩니다.

```bash
python3 predict.py --i=2020311640 --l=b --b=25 --q="한식 얼큰한 육류"
```

| Argument | 설명 | 코드 내 사용 방식 |
| - | - | - |
| `--i` (User ID) | **필수:** 추천을 요청하는 사용자의 고유 ID. | **CF 점수 예측**의 주체로 사용되며, `user_data.csv`에서 사용자의 기존 선호도 조회에 사용됩니다. |
| `--l` (위치) | **필수:** 사용자의 현재 위치 코드 (`s`:성균관대역, `b`:후문, `n`:북문, `f`:정문). | **거리 점수 계산**에 사용되어 후보 메뉴에 대한 `실시간 접근성 점수`를 도출합니다. |
| `--b` (예산) | **필수:** 사용자의 최대 예산 (단위: 천 원). | **하드 필터링**에 직접 사용되어 예산을 초과하는 메뉴를 후보 목록에서 제외합니다. |
| `--q` (쿼리) | **선택:** 사용자의 즉각적인 선호도를 반영하는 키워드. | **CB 점수 계산** 시, `user_data.preference`와 **결합**되어 최종 콘텐츠 선호도 벡터를 구성합니다. |

**출력 내용 설명**
`predict.py`는 하드 필터링(예산, 알레르기)을 통과한 메뉴들을 `MLP` 모델에 입력하여 **예상 평점** (`Predicted_Rating`)을 계산한 다음, 그 결과를 정렬하여 Top N개의 메뉴를 출력합니다.
* 출력 항목: 메뉴명, 식당명, 가격, 예상 평점.
* 미평가 필터링: 현재 추천 로직은 사용자가 **이전에 평점을 남기지 않은 항목만** 추천 목록에 표시합니다.
    * (미평가 필터링은 추후 삭제하여 이전에 유저가 평가했던 항목도 추천 목록에 뜨도록 수정할 수 있습니다.)