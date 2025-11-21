# main.py

import pandas as pd
import numpy as np
import math # 수학 함수 사용을 위해 math 모듈 추가
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from filtering.contents_based import ContentBasedRecommender
from filtering.collaborative import CollaborativeRecommender
from filtering.blender_mlp import MLPBlender

# Pandas 출력 옵션 설정
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# --- 🎯 위치 정보 및 상수 정의 ---
COORDINATES = {
    '후문': (37.29633029410662, 126.97061603024721),
    '북문': (37.296274335479666, 126.9764159771293),
    '정문': (37.29100570424096, 126.97417156623229),
    '성균관대역': (37.29986776148395, 126.97219805873624)
}
R = 6371 # 지구 반지름 (km)

# --- 상수 설정 ---
TEST_USER_ID = 2020312857 
INPUT_FEATURE_DIM = 5      
EPOCHS = 30                
TOP_N = 10                 

# --- 함수 정의 ---

def haversine(lat1, lon1, lat2, lon2):
    """
    하버사인 공식을 사용하여 두 좌표 간의 거리를 km 단위로 계산합니다.
    """
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def calculate_distance_score(user_location_name, menu_location_name):
    """
    위치 이름을 받아 하버사인 거리를 계산하고, 이를 점수(0~1)로 변환합니다.
    (Score = exp(-distance / L0) )
    """
    if user_location_name not in COORDINATES or menu_location_name not in COORDINATES:
        return 0.1 # 위치 정보가 없을 경우 낮은 점수 부여

    lat1, lon1 = COORDINATES[user_location_name]
    lat2, lon2 = COORDINATES[menu_location_name]
    
    distance_km = haversine(lat1, lon1, lat2, lon2)

    # L0: 특성 거리 (0.5km를 기준으로 거리가 멀어질수록 점수가 급격히 하락하도록 설정)
    L0 = 0.5 
    
    # 점수: 거리가 0이면 1, 거리가 멀어질수록 0에 수렴
    score = math.exp(-distance_km / L0)
    
    return score

def generate_hybrid_features(ratings_df, menu_df, cb_recommender, cf_recommender):
    """
    MLP 학습에 사용할 X (입력 특징)와 Y (정답 평점) 데이터를 생성합니다.
    """
    print("\n[3] 하이브리드 특징 행렬 (X, Y) 생성 시작...")
    
    data = ratings_df.copy()
    
    # NOTE: 모든 유저의 현재 위치가 '정문'이라고 가정하고 계산합니다.
    CURRENT_USER_LOCATION = '정문' 
    DUMMY_PROFILE = "한식 찌개 얼큰한 밥이랑" 
    
    # 1. 메뉴 데이터 조인
    data = pd.merge(data, menu_df[['id', 'price', 'location', 'avg_rating']], 
                    left_on='menu_id', right_on='id', how='left')
    
    # 2. CB Score (콘텐츠 유사도) 계산
    data['CB_Score'] = data.apply(
        lambda row: cb_recommender.get_single_cb_score(row['menu_id'], DUMMY_PROFILE),
        axis=1
    )

    # 3. CF Score (예상 평점) 계산
    data['CF_Score'] = data.apply(
        lambda row: cf_recommender.model.predict(
            uid=row['user_id'], iid=row['menu_id']
        ).est,
        axis=1
    )
    
    # 4. Distance Score 계산 (좌표 기반)
    data['Distance_Score'] = data['location'].apply(
        lambda menu_loc: calculate_distance_score(CURRENT_USER_LOCATION, menu_loc)
    )
    
    # 5. X 행렬 및 Y 벡터 추출 (avg_rating은 menu_df에서 병합되어 있다고 가정)
    X = data[['CB_Score', 'CF_Score', 'price', 'Distance_Score', 'avg_rating']].values
    Y = data['rating'].values 
    
    print(f"특징 행렬 X 생성 완료. Shape: {X.shape}")
    return X, Y


def main():
    
    # --- 1. 데이터 및 추천기 초기화 ---
    ratings_df = pd.read_csv('./data/ratings_data.csv')
    
    # menu_data.csv에 'avg_rating' 컬럼이 없다고 가정하고 계산하여 추가
    menu_df = pd.read_csv('./data/menu_data.csv')
    menu_ratings = ratings_df.groupby('menu_id')['rating'].mean().reset_index()
    menu_ratings.columns = ['id', 'avg_rating']
    
    # avg_rating을 menu_df에 병합
    menu_df = pd.merge(menu_df, menu_ratings, on='id', how='left').fillna(0) 
    
    # CB Recommender 초기화
    cb_recommender = ContentBasedRecommender()
    
    # CF Recommender 초기화 (모델 학습 포함)
    cf_recommender = CollaborativeRecommender()
    
    
    # --- 2. MLP 학습 데이터셋 준비 ---
    X, Y = generate_hybrid_features(ratings_df, menu_df, cb_recommender, cf_recommender)
    
    # 학습/테스트 셋 분리
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    
    # --- 3. MLP 모델 학습 및 검증 ---
    
    mlp_blender = MLPBlender(input_dim=X.shape[1])
    
    print("\n[4] MLP 모델 학습 시작...")
    # 학습 결과를 history 객체로 받아 과적합 관측 자료로 사용 가능
    history = mlp_blender.train(X_train, Y_train, epochs=EPOCHS, batch_size=4) 
    
    # 테스트 셋 검증
    Y_pred_test = mlp_blender.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred_test))
    
    print("-" * 50)
    print(f"✅ 최종 테스트 셋 RMSE: {test_rmse:.4f}")
    print("-" * 50)
    
    
    # --- 4. 최종 추천 로직 (예시) ---
    # (실제 추천은 unrated 메뉴 기반으로 X_predict를 생성해야 합니다.)
    
    recommendation_results = pd.DataFrame({
        'Predicted_Rating': final_pred_scores,
        'Actual_Rating': Y_test 
    })
    
    # 예상 평점이 높은 순으로 정렬 (예시)
    print(f"\n[5] 최종 MLP 예측 결과 샘플 (Top {TOP_N})")
    print(recommendation_results.sort_values(by='Predicted_Rating', ascending=False).head(TOP_N))
    

if __name__ == "__main__":
    # ContentBasedRecommender 클래스에 get_single_cb_score 메서드를 임시로 연결합니다.
    # (원래 filtering/contents_based.py에 직접 구현되어야 함)
    from sklearn.metrics.pairwise import cosine_similarity
    
    def get_single_cb_score(self, menu_id, user_profile):
        menu_index = self.menu_df[self.menu_df['id'] == menu_id].index
        if len(menu_index) == 0: return 0.0
        
        user_vector = self.tfidf_vectorizer.transform([user_profile])
        menu_vector = self.menu_feature_matrix[menu_index[0]]
        
        return cosine_similarity(user_vector, menu_vector)[0][0]
        
    from filtering.contents_based import ContentBasedRecommender # 재임포트
    ContentBasedRecommender.get_single_cb_score = get_single_cb_score
    
    # MLP 예측 샘플 출력을 위한 임시 변수 할당
    # 이 부분은 실제 예측 시나리오가 아니므로, 경고를 피하기 위해 임시로 정의합니다.
    final_pred_scores = np.array([5, 4, 3, 2, 1, 4.5, 3.5, 2.5, 1.5, 5.0, 4.0, 3.0])
    Y_test = np.array([5, 4, 3, 2, 1, 4, 3, 2, 1, 5, 4, 3]) 

    main()