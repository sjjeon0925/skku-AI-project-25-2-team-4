import pandas as pd
import numpy as np
import math
import argparse # 사용자 입력 처리를 위해 argparse 추가
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from filtering.contents_based import ContentBasedRecommender
from filtering.collaborative import CollaborativeRecommender
from filtering.blender_mlp import MLPBlender

# --- 🎯 위치 정보 및 상수 정의 ---
COORDINATES = {
    's': (37.29986776148395, 126.97219805873624), # 성균관대역
    'b': (37.29633029410662, 126.97061603024721), # 후문 (Back gate)
    'n': (37.296274335479666, 126.9764159771293), # 북문 (North gate)
    'f': (37.29100570424096, 126.97417156623229), # 정문 (Front gate)
}
R = 6371 # 지구 반지름 (km)

# --- 데이터 파일 경로 ---
DATA_PATHS = {
    'menu': './data/menu_data.csv',
    'rest': './data/rest_data.csv',
    'user': './data/user_data.csv',
    'rating': './data/rating_data.csv',
}

# --- 지리 및 유틸리티 함수 ---

def haversine(lat1, lon1, lat2, lon2):
    """하버사인 공식을 사용하여 두 좌표 간의 거리를 km 단위로 계산합니다."""
    # ... (기존 하버사인 함수 로직) ...
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_distance_score(user_loc_char, rest_lat, rest_lon):
    """현재 사용자 위치와 식당 좌표 간의 거리 점수를 계산합니다."""
    if user_loc_char not in COORDINATES: return 0.0

    user_lat, user_lon = COORDINATES[user_loc_char]
    distance_km = haversine(user_lat, user_lon, rest_lat, rest_lon)

    L0 = 0.5 # 특성 거리
    score = math.exp(-distance_km / L0)
    return score

def get_cb_preference(user_id, query_str):
    """
    CB Score 계산에 사용할 최종 선호도 문자열을 결정합니다.
    쿼리 우선순위 로직을 구현합니다.
    """
    user_df = pd.read_csv(DATA_PATHS['user'])
    user_pref = user_df[user_df['user_id'] == user_id]['preference'].iloc[0]
    
    if pd.isna(query_str) or query_str == "":
        # 1. 쿼리가 없는 경우: user_data preference 사용
        return user_pref
    
    # 2. 쿼리가 있는 경우 (Gemini 호출 없이 임시 결합 로직 사용)
    # (실제로는 Gemini 호출을 통해 쿼리를 핵심 키워드로 변환 후 결합)
    
    # 임시: 쿼리와 기존 선호도를 공백으로 결합하여 사용 (쿼리에 높은 가중치 부여 효과)
    # 예: user_pref + query_str
    
    # 쿼리에 하드 필터 요소(예산, 치즈 등)가 포함되어 있다면, 
    # 이는 필터링에 사용되어야 하지만, 여기서는 CB Score 계산을 위해 문자열을 결합합니다.
    return user_pref + " " + query_str


# --- 데이터 로드 및 특징 생성 (MLP 학습용) ---

def generate_hybrid_features(user_loc_char, cb_recommender, cf_recommender):
    """
    MLP 학습에 사용할 X (입력 특징)와 Y (정답 평점) 데이터를 생성합니다.
    (MLP 학습 데이터는 rating_data를 기반으로 하며, 예측이 아닌 정답 평점 Y를 타겟으로 함)
    """
    print("\n[3] 하이브리드 특징 행렬 (X, Y) 생성 시작...")
    
    # 1. 데이터 로드
    rating_df = pd.read_csv(DATA_PATHS['rating'])
    menu_df = pd.read_csv(DATA_PATHS['menu'])
    rest_df = pd.read_csv(DATA_PATHS['rest'])
    user_df = pd.read_csv(DATA_PATHS['user'])
    
    # 2. 데이터 병합 (rating_df -> menu_df -> rest_df)
    data = pd.merge(rating_df, menu_df[['menu_id', 'rest_id', 'price', 'features']], on='menu_id', how='left')
    data = pd.merge(data, rest_df[['rest_id', 'Latitude', 'Longitude', 'rating']], on='rest_id', how='left')
    
    # 3. 특징 계산
    data['CB_Score'] = data.apply(
        lambda row: cb_recommender.get_single_cb_score(row['menu_id'], row['user_id'], user_df),
        axis=1
    )
    
    data['CF_Score'] = data.apply(
        lambda row: cf_recommender.model.predict(
            uid=row['user_id'], iid=row['menu_id']
        ).est,
        axis=1
    )
    
    # NOTE: 학습 데이터셋은 "평가 당시 위치"를 거리 계산의 유저 위치로 사용해야 정확하지만,
    # 현재는 간단화를 위해 모든 유저의 현재 위치('정문'이라고 가정)와 비교합니다.
    # 하지만 사용자 요구사항에 따라 '평가 당시 위치'를 활용합니다.
    location_map = {v: k for k, v in [('성균관대역', 's'), ('정문', 'f'), ('후문', 'b'), ('북문', 'n')]} # 임시 매핑
    data['Distance_Score'] = data.apply(
        lambda row: calculate_distance_score(
            location_map.get(row['location'], 'f'), # 평가 당시 위치 (문자열->코드)
            row['Latitude'], row['Longitude']
        ),
        axis=1
    )
    
    # 4. 정규화 및 X, Y 추출
    # Price와 Avg. Rating 정규화를 위한 Scaler는 MLPBlender에서 처리됨을 가정하고, raw data를 넘김
    X = data[['CB_Score', 'CF_Score', 'price', 'Distance_Score', 'rating']].values # rating은 rest_data의 rating
    Y = data['rating_x'].values # rating_x는 rating_data의 rating (정답 평점)
    
    print(f"특징 행렬 X 생성 완료. Shape: {X.shape}")
    return X, Y


# --- 메인 실행 함수 ---

def main():
    
    parser = argparse.ArgumentParser(description="SKKU Menu Hybrid Recommendation Engine")
    parser.add_argument('--i', type=int, required=True, help='User ID (e.g., 2020)')
    parser.add_argument('--l', type=str, required=True, choices=COORDINATES.keys(), help='Current Location Code (s, b, n, f)')
    parser.add_argument('--b', type=int, default=10, help='Budget (in thousand KRW, e.g., 10 for 10,000 KRW)')
    parser.add_argument('--q', type=str, default="", help='Optional query for content filtering (e.g., "치즈가 들어간 메뉴")')
    args = parser.parse_args()
    
    USER_ID = args.i
    USER_LOC_CHAR = args.l
    USER_BUDGET = args.b * 1000
    USER_QUERY = args.q

    print("-" * 60)
    print(f"사용자 요청: ID={USER_ID}, 위치={USER_LOC_CHAR}, 예산={USER_BUDGET}원, 쿼리='{USER_QUERY}'")
    print("-" * 60)

    # 1. 추천기 초기화
    cb_recommender = ContentBasedRecommender()
    cf_recommender = CollaborativeRecommender()
    
    # 2. MLP 학습 데이터셋 준비 (모든 rating_data 기반)
    X, Y = generate_hybrid_features(USER_LOC_CHAR, cb_recommender, cf_recommender)
    
    # 3. 학습/테스트 셋 분리
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # 4. MLP 모델 학습 (INPUT_FEATURE_DIM = 5)
    mlp_blender = MLPBlender(input_dim=X.shape[1])
    print("\n[5] MLP 모델 학습 시작...")
    mlp_blender.train(X_train, Y_train, epochs=30, batch_size=4) 
    
    # 5. 최종 추천 후보군 생성 (Hard Filtering)
    menu_df = pd.read_csv(DATA_PATHS['menu'])
    user_df = pd.read_csv(DATA_PATHS['user'])
    
    # 5-1. 알레르기 및 예산 필터링
    user_allergy = user_df[user_df['user_id'] == USER_ID]['allergy'].iloc[0]
    
    candidate_df = menu_df[(menu_df['price'] <= USER_BUDGET) & (~menu_df['features'].str.contains(user_allergy, na=False))].copy()
    
    # 6. 최종 예측 특징 생성 (현재 위치, 쿼리 기반)
    # MLP 예측에 필요한 X_predict 행렬을 생성해야 함
    # 이 부분은 'generate_hybrid_features'와 유사하나, 유저 ID에 대한 모든 *unrated* 메뉴에 대해 수행되어야 함 (생략하고 샘플 예측)
    
    # 7. (샘플 예측) 테스트 셋에 대해 예측 및 출력
    Y_pred_test = mlp_blender.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred_test))
    
    print("-" * 60)
    print(f"✅ 최종 MLP 테스트 셋 RMSE: {test_rmse:.4f}")
    
    # 8. 최종 추천 결과 출력 (예시)
    recommendation_results = pd.DataFrame({
        'Predicted_Rating': Y_pred_test,
        'Actual_Rating': Y_test 
    })
    
    print(f"\n[6] 최종 MLP 예측 결과 샘플 (Top 10)")
    print(recommendation_results.sort_values(by='Predicted_Rating', ascending=False).head(10))


if __name__ == "__main__":
    # ContentBasedRecommender 클래스에 get_single_cb_score 메서드를 임시로 연결합니다.
    from sklearn.metrics.pairwise import cosine_similarity
    
    def get_single_cb_score(self, menu_id, user_id, user_df):
        # 1. User Preference 가져오기 (CB Score 계산 시에는 쿼리 로직 없이 user_data 기준)
        user_pref = user_df[user_df['user_id'] == user_id]['preference'].iloc[0]
        
        # 2. TF-IDF 벡터 생성 및 유사도 계산
        menu_index = self.menu_df[self.menu_df['menu_id'] == menu_id].index
        if len(menu_index) == 0: return 0.0
        
        user_vector = self.tfidf_vectorizer.transform([user_pref])
        menu_vector = self.menu_feature_matrix[menu_index[0]]
        
        return cosine_similarity(user_vector, menu_vector)[0][0]
        
    # Class Linkage
    from filtering.contents_based import ContentBasedRecommender
    ContentBasedRecommender.get_single_cb_score = get_single_cb_score
    
    main()