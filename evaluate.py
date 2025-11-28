import pandas as pd
import numpy as np
import os
import tensorflow as tf

# 모듈 임포트
from filtering.contents_based import ContentBasedRecommender
from filtering.collaborative import CollaborativeRecommender
from filtering.blender_mlp import MLPBlender
from filtering.graph_model import GraphRecommender
from utils import (
    DATA_PATHS, calculate_distance_score, IS_BASELINE,
    MLP_MODEL_PATH, SCALER_PATH, GRAPH_MODEL_PATH, INPUT_FEATURE_DIM
)

def evaluate_recall(user_id, true_menu_ids, top_k=10):
    print(f"\n--- 평가 시작 (User: {user_id}) ---")
    print(f"MODE: {'BASELINE' if IS_BASELINE else 'GNN'}")
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

    if not IS_BASELINE:
        gnn_recommender = GraphRecommender(DATA_PATHS['rating'], DATA_PATHS['menu'])
        if os.path.exists(GRAPH_MODEL_PATH):
            gnn_recommender.load_model(GRAPH_MODEL_PATH)
            print("GNN 모델 로드 완료.")
        else:
            print("[Warning] GNN 모델 파일이 없습니다.")

    # MLP 로드
    mlp_blender = MLPBlender(input_dim=INPUT_FEATURE_DIM)
    if os.path.exists(MLP_MODEL_PATH):
        mlp_blender.model = tf.keras.models.load_model(MLP_MODEL_PATH)
        mlp_blender.load_scaler(SCALER_PATH)
    else:
        print(f"MLP 모델 파일({MLP_MODEL_PATH})이 없습니다. train.py를 먼저 실행하세요.")
        return

    # 3. 전체 메뉴 후보군 생성 (All Candidates)
    candidate_df = menu_df.copy()
    
    # Feature 생성
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
    if not IS_BASELINE and gnn_recommender:
        candidate_df['Graph_Score'] = candidate_df['menu_id'].apply(
            lambda x: gnn_recommender.get_graph_score(user_id, x)
        )
    
    # D. Meta Scores (Price, Distance, Rating)
    candidate_df = pd.merge(candidate_df, rest_df[['rest_id', 'Latitude', 'Longitude', 'rating']], on='rest_id', how='left')
    candidate_df.rename(columns={'rating': 'Avg_Rating'}, inplace=True)
    candidate_df['Avg_Rating'] = candidate_df['Avg_Rating'].fillna(3.0)

    # Distance Score 계산 (utils.py 활용, 기준 위치: 후문 'b')
    TARGET_LOCATION = 'b' # 후문 기준
    candidate_df['Distance_Score'] = candidate_df.apply(
        lambda row: calculate_distance_score(TARGET_LOCATION, row['Latitude'], row['Longitude']), 
        axis=1
    )
    
    # 4. MLP 예측
    if IS_BASELINE:
        X_eval = candidate_df[['CB_Score', 'CF_Score', 'price', 'Distance_Score', 'Avg_Rating']].values
    else:
        X_eval = candidate_df[['CB_Score', 'CF_Score', 'Graph_Score', 'price', 'Distance_Score', 'Avg_Rating']].values
        
    candidate_df['Predicted_Rating'] = mlp_blender.predict(X_eval)
    
    # 5. 결과 확인
    top_n_df = candidate_df.sort_values(by='Predicted_Rating', ascending=False).head(top_k)
    recommended_ids = top_n_df['menu_id'].tolist()
    
    hits = set(true_menu_ids) & set(recommended_ids)
    hit_count = len(hits)
    total_true = len(true_menu_ids)
    recall = hit_count / total_true if total_true > 0 else 0.0
    
    print(f"\n[평가 결과 (Mode: {'Baseline' if IS_BASELINE else 'GNN'})]")
    print(f"- Recall@{top_k}: {recall * 100:.2f}% ({hit_count}/{total_true})")

if __name__ == "__main__":
    # 테스트용 데이터 설정 (예시)
    TEST_USER_ID = 2020311640
    TRUE_MENU_IDS = [84, 54, 71, 46,45,40,23,81,11,1,19,31,74,34,91,5,77,78,13] 
    
    evaluate_recall(TEST_USER_ID, TRUE_MENU_IDS, top_k=len(TRUE_MENU_IDS))