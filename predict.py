import pandas as pd
import numpy as np
import argparse
import os
import joblib 
from sklearn.metrics import mean_squared_error
from filtering.contents_based import ContentBasedRecommender
from filtering.collaborative import CollaborativeRecommender
from filtering.blender_mlp import MLPBlender
from tensorflow.keras.models import load_model
from utils import DATA_PATHS, COORDINATES, calculate_distance_score, get_cb_preference
from sklearn.metrics.pairwise import cosine_similarity # CB Score 계산용

# --- 상수 설정 ---
INPUT_FEATURE_DIM = 5 
MODEL_PATH = 'model/final_mlp_model.keras'
SCALER_PATH = 'model/scaler.joblib' 

LOCATION_NAME = {
    's': '성균관대역',
    'b': '후문',
    'n': '북문',
    'f': '정문'
}

def get_unrated_menu_ids(user_id, all_menu_ids, ratings_df):
    """
    특정 유저가 아직 평가하지 않은 메뉴 ID 리스트를 반환합니다.
    """
    rated_menus = ratings_df[ratings_df['user_id'] == user_id]['menu_id'].unique()
    return list(set(all_menu_ids) - set(rated_menus))


def generate_prediction_features_predict(candidate_df, user_id, user_loc_char, final_user_pref, cb_recommender, cf_recommender, rest_df):
    """
    실시간 예측을 위한 특징 행렬 X_predict를 생성합니다. (라이브 유저 입력 반영)
    """
    
    X_predict_data = candidate_df.copy()

    # NOTE: predict 시에는 USER_QUERY가 반영된 final_user_pref 문자열을 직접 사용합니다.
    
    # 1. 특징 계산 및 데이터 병합
    X_predict_data = pd.merge(X_predict_data, rest_df[['rest_id', 'Latitude', 'Longitude', 'rating']], on='rest_id', how='left', suffixes=('_menu', '_rest'))

    # CB Score (현재 쿼리 기반)
    X_predict_data['CB_Score'] = X_predict_data['menu_id'].apply(
        lambda menu_id: cb_recommender.get_single_cb_score(menu_id, final_user_pref) 
    )

    # CF Score (Collaborative Filtering 예상 평점)
    X_predict_data['CF_Score'] = X_predict_data['menu_id'].apply(
        lambda menu_id: cf_recommender.model.predict(
            uid=user_id, iid=menu_id
        ).est
    )
    
    # Distance Score (현재 위치 기반)
    X_predict_data['Distance_Score'] = X_predict_data.apply(
        lambda row: calculate_distance_score(
            user_loc_char, row['Latitude'], row['Longitude']
        ),
        axis=1
    )
    
    # Avg. Restaurant Rating (식당 평점)
    X_predict_data['Avg_Rating'] = X_predict_data['rating']

    # 2. MLP 예측을 위한 X_predict 행렬 추출
    X_predict = X_predict_data[['CB_Score', 'CF_Score', 'price', 'Distance_Score', 'Avg_Rating']].values
    
    return X_predict, X_predict_data


def main():
    parser = argparse.ArgumentParser(description="SKKU Menu Hybrid Recommendation Predictor")
    parser.add_argument('--i', type=int, required=True, help='User ID (e.g., 2020)')
    parser.add_argument('--l', type=str, required=True, choices=COORDINATES.keys(), help='Current Location Code (s, b, n, f)')
    parser.add_argument('--b', type=int, default=10, help='Budget (in thousand KRW, e.g., 10 for 10,000 KRW)')
    parser.add_argument('--q', type=str, default="", help='Optional query for content filtering')
    args = parser.parse_args()
    
    USER_ID = args.i
    USER_LOC_CHAR = args.l
    USER_BUDGET = args.b * 1000
    USER_QUERY = args.q

    # 1. 모델/스케일러 로드 및 추천기 초기화 확인
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: 학습된 모델 또는 Scaler 파일이 없습니다. train.py를 먼저 실행하세요.")
        return

    # Custom Metric 정의 (로드 시 필수)
    def root_mean_squared_error(y_true, y_pred):
         return np.sqrt(mean_squared_error(y_true, y_pred))

    # MLP/Scaler 로드
    mlp_blender = MLPBlender(input_dim=INPUT_FEATURE_DIM)
    mlp_blender.model = load_model(MODEL_PATH, custom_objects={'root_mean_squared_error': root_mean_squared_error})
    mlp_blender.load_scaler(SCALER_PATH) # [핵심] Scaler 로드

    # 추천기 로드 및 데이터 로드
    cb_recommender = ContentBasedRecommender(data_path=DATA_PATHS['menu'])
    cf_recommender = CollaborativeRecommender(ratings_path=DATA_PATHS['rating'], menu_path=DATA_PATHS['menu'])
    
    menu_df = pd.read_csv(DATA_PATHS['menu'])
    user_df = pd.read_csv(DATA_PATHS['user'])
    rest_df = pd.read_csv(DATA_PATHS['rest'])
    ratings_df = pd.read_csv(DATA_PATHS['rating']) # unrated 메뉴를 찾기 위해 로드
    
    # 2. Hard Filtering (유저 입력 사용 O)
    user_row = user_df[user_df['user_id'] == USER_ID]
    if user_row.empty:
        print(f"Error: User ID {USER_ID} not found in user_data.")
        return
        
    user_allergy = user_row['allergy'].iloc[0]
    
    # 알레르기 값 존재 여부 확인 후 분기 처리
    if pd.isna(user_allergy):
        # 알레르기 없으면: 예산만 필터링
        candidate_df = menu_df[menu_df['price'] <= USER_BUDGET].copy()
    else:
        # 알레르기 있으면: 예산 + 알레르기 필터링
        candidate_df = menu_df[
            (menu_df['price'] <= USER_BUDGET) & 
            (~menu_df['features'].str.contains(user_allergy, na=False))
        ].copy()
    
    # 2-2. 미평가 메뉴 필터링 (CF의 목적을 위해)
    unrated_menu_ids = get_unrated_menu_ids(USER_ID, menu_df['menu_id'], ratings_df)
    candidate_df = candidate_df[candidate_df['menu_id'].isin(unrated_menu_ids)]
    
    print("-" * 60)
    print(f"사용자 요청: ID={USER_ID}, 위치={LOCATION_NAME[USER_LOC_CHAR]}, 예산={USER_BUDGET}원")
    print(f"하드 필터링 및 미평가 필터링 통과 후보 수: {candidate_df.shape[0]}개")
    print("-" * 60)

    if candidate_df.empty:
        print("❌ 추천 가능한 메뉴가 없습니다. (예산, 알레르기 또는 미평가 항목 없음)")
        return

    # 3. 최종 예측 특징 행렬 생성 및 예측 수행
    final_user_pref = get_cb_preference(USER_ID, USER_QUERY) # 쿼리 통합 선호도 문자열
    
    X_predict_raw, result_df = generate_prediction_features_predict(
        candidate_df, USER_ID, USER_LOC_CHAR, final_user_pref, cb_recommender, cf_recommender, rest_df
    )
    
    # 4. MLP 예측
    final_pred_scores = mlp_blender.predict(X_predict_raw)
    result_df['Predicted_Rating'] = final_pred_scores
    
    # 5. 결과 출력
    final_ranking = result_df.sort_values(by='Predicted_Rating', ascending=False)
    
    print("\n✅ 최종 MLP 예측 결과 (Top 10)")
    final_ranking = pd.merge(final_ranking, rest_df[['rest_id', 'rest_name']], on='rest_id', how='left')
    
    # 최종 결과 출력: 메뉴명, 식당명, 가격, 예상 평점
    print(final_ranking[['menu', 'rest_name', 'price', 'Predicted_Rating']].head(10))


if __name__ == "__main__":
    
    main()