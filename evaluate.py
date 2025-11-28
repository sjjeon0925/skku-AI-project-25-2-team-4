import pandas as pd
import numpy as np
import os
import tensorflow as tf

# 모듈 임포트
from filtering.contents_based import ContentBasedRecommender
from filtering.collaborative import CollaborativeRecommender
from filtering.blender_mlp import MLPBlender
from utils import calculate_distance_score, DATA_PATHS 

# GNN 모듈은 파일이 있을 때만 임포트
try:
    from filtering.graph_model import GraphRecommender
    USE_GNN = True
except ImportError:
    USE_GNN = False

# 모델 파일 경로
MLP_MODEL_PATH = 'model/final_mlp_model.keras'
SCALER_PATH = 'model/scaler.joblib'
GRAPH_MODEL_PATH = 'model/gnn_model.pth'

def evaluate_recall(user_id, true_menu_ids, top_k=10):
    """
    특정 사용자의 정답 메뉴 리스트(true_menu_ids)를 모델이 얼마나 잘 맞히는지 평가합니다.
    """
    print(f"\n--- 평가 시작 (User: {user_id}) ---")
    print(f"정답 메뉴 리스트(Ground Truth): {true_menu_ids}")

    # 1. 데이터 로드
    menu_df = pd.read_csv(DATA_PATHS['menu'])
    rest_df = pd.read_csv(DATA_PATHS['rest'])
    ratings_df = pd.read_csv(DATA_PATHS['rating'])
    user_df = pd.read_csv(DATA_PATHS['user'])

    # 2. 모델 로드
    print("모델 로딩 중...")
    cb_recommender = ContentBasedRecommender(DATA_PATHS['menu'])
    cf_recommender = CollaborativeRecommender(DATA_PATHS['rating'], DATA_PATHS['menu'])
    
    gnn_recommender = None
    input_dim = 5 # 기본 5개

    if USE_GNN and os.path.exists(GRAPH_MODEL_PATH):
        gnn_recommender = GraphRecommender(DATA_PATHS['rating'], DATA_PATHS['menu'])
        gnn_recommender.load_model(GRAPH_MODEL_PATH)
        print("GNN 모델 로드 완료. (GNN 모드)")
        input_dim = 6
    else:
        print("GNN 모델을 사용하지 않거나 파일이 없습니다. (Baseline 모드)")

    # MLP 로드
    mlp_blender = MLPBlender(input_dim=input_dim)
    if os.path.exists(MLP_MODEL_PATH):
        mlp_blender.model = tf.keras.models.load_model(MLP_MODEL_PATH)
        mlp_blender.load_scaler(SCALER_PATH)
    else:
        print("MLP 모델 파일이 없습니다. train.py를 먼저 실행하세요.")
        return

    # 3. 전체 메뉴 후보군 생성 (All Candidates)
    # [중요] 평가를 위해 '이미 먹은 메뉴' 필터링 없이 전체를 복사합니다.
    candidate_df = menu_df.copy()
    
    # --- Feature 생성 ---
    
    # A. CB Score
    user_pref = user_df[user_df['user_id'] == user_id]['preference'].iloc[0]
    candidate_df['CB_Score'] = candidate_df['menu_id'].apply(
        lambda x: cb_recommender.get_single_cb_score(x, user_pref)
    )
    
    # B. CF Score
    cf_scores = cf_recommender.get_predicted_scores(user_id, candidate_df['menu_id'].tolist())
    cf_score_map = {mid: score for mid, score in cf_scores}
    candidate_df['CF_Score'] = candidate_df['menu_id'].map(cf_score_map)
    
    # C. GNN Score (Optional)
    if gnn_recommender:
        candidate_df['Graph_Score'] = candidate_df['menu_id'].apply(
            lambda x: gnn_recommender.get_graph_score(user_id, x)
        )
    
    # D. Meta Scores (Price, Distance, Rating)
    # 식당 정보(위치 포함) 병합
    candidate_df = pd.merge(candidate_df, rest_df[['rest_id', 'Latitude', 'Longitude', 'rating']], on='rest_id', how='left')
    candidate_df.rename(columns={'rating': 'Avg_Rating'}, inplace=True)
    candidate_df['Avg_Rating'] = candidate_df['Avg_Rating'].fillna(3.0)

    # Distance Score 계산 (utils.py 활용, 기준 위치: 후문 'b')
    TARGET_LOCATION = 'b' # 후문
    candidate_df['Distance_Score'] = candidate_df.apply(
        lambda row: calculate_distance_score(TARGET_LOCATION, row['Latitude'], row['Longitude']), 
        axis=1
    )
    
    # 4. MLP 예측
    # Feature 순서 중요: 학습 때와 동일해야 함
    if gnn_recommender:
        X_eval = candidate_df[['CB_Score', 'CF_Score', 'Graph_Score', 'price', 'Distance_Score', 'Avg_Rating']].values
    else:
        X_eval = candidate_df[['CB_Score', 'CF_Score', 'price', 'Distance_Score', 'Avg_Rating']].values
        
    candidate_df['Predicted_Rating'] = mlp_blender.predict(X_eval)
    
    # 5. 결과 정렬 및 비교
    top_n_df = candidate_df.sort_values(by='Predicted_Rating', ascending=False).head(top_k)
    recommended_ids = top_n_df['menu_id'].tolist()
    
    print(f"\n[모델 추천 Top {top_k}]")
    for idx, row in top_n_df.iterrows():
        print(f"{row['menu']} (ID: {row['menu_id']}) - 예상점수: {row['Predicted_Rating']:.4f}")
        
    # 6. 적중률(Recall) 계산
    hits = set(true_menu_ids) & set(recommended_ids)
    hit_count = len(hits)
    total_true = len(true_menu_ids)
    recall = hit_count / total_true if total_true > 0 else 0.0
    
    print(f"\n[평가 결과]")
    print(f"- 정답 메뉴: {true_menu_ids}")
    print(f"- 맞춘 메뉴: {list(hits)}")
    print(f"- Recall@{top_k}: {recall * 100:.2f}% ({hit_count}/{total_true})")

if __name__ == "__main__":
    # 테스트용 데이터 설정 (예시)
    TEST_USER_ID = 2020311640
    TRUE_MENU_IDS = [54, 84, 1] 
    
    evaluate_recall(TEST_USER_ID, TRUE_MENU_IDS, top_k=10)