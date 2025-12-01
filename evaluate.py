import pandas as pd
import numpy as np
import os
import tensorflow as tf

# [요청하신 임포트 경로 적용]
from filtering.contents_based import ContentBasedRecommender
from filtering.collaborative import CollaborativeRecommender
from filtering.blender_mlp import MLPBlender
from filtering.graph_model import GraphRecommender
from utils import (
    DATA_PATHS, calculate_distance_score, IS_BASELINE,
    MLP_MODEL_PATH, SCALER_PATH, GRAPH_MODEL_PATH, INPUT_FEATURE_DIM
)

# [진단용 플래그] True: GNN 점수로만 랭킹 산정 (진단용), False: MLP 사용 (기본)
TEST_PURE_GNN = False 

def load_models():
    """모델들을 메모리에 한 번만 로드하여 반환합니다."""
    print("모델 및 데이터 로딩 중...")
    
    # 데이터 로드
    menu_df = pd.read_csv(DATA_PATHS['menu'])
    rest_df = pd.read_csv(DATA_PATHS['rest'])
    ratings_df = pd.read_csv(DATA_PATHS['rating'])
    user_df = pd.read_csv(DATA_PATHS['user'])
    
    # 추천기 초기화
    cb_recommender = ContentBasedRecommender(DATA_PATHS['menu'])
    cf_recommender = CollaborativeRecommender(DATA_PATHS['rating'], DATA_PATHS['menu'])
    
    gnn_recommender = None

    if not IS_BASELINE:
        gnn_recommender = GraphRecommender(DATA_PATHS['rating'], DATA_PATHS['menu'])
        if os.path.exists(GRAPH_MODEL_PATH):
            gnn_recommender.load_model(GRAPH_MODEL_PATH)
            print("✅ GNN 모델 로드 완료.")
        else:
            print("⚠️ [Warning] GNN 모델 파일이 없습니다.")

    # MLP 로드
    mlp_blender = MLPBlender(input_dim=INPUT_FEATURE_DIM)
    if os.path.exists(MLP_MODEL_PATH):
        mlp_blender.model = tf.keras.models.load_model(MLP_MODEL_PATH)
        mlp_blender.load_scaler(SCALER_PATH)
        print("✅ MLP 모델 로드 완료.")
    else:
        print(f"❌ Error: MLP 모델 파일({MLP_MODEL_PATH})이 없습니다.")
        # [수정] 반환 값 개수 맞춤 (8개)
        return None, None, None, None, None, None, None, None

    return menu_df, rest_df, ratings_df, user_df, cb_recommender, cf_recommender, gnn_recommender, mlp_blender

def evaluate_single_user(user_id, true_menu_ids, models_data, top_k=10, verbose=False):
    """단일 사용자에 대한 Recall@K를 계산합니다."""
    (menu_df, rest_df, ratings_df, user_df, cb, cf, gnn, mlp) = models_data
    
    # 1. 사용자 선호도(Preference) 가져오기
    user_row = user_df[user_df['user_id'].astype(str) == str(user_id)]
    if user_row.empty:
        if verbose: print(f"User {user_id} 정보 없음 - Skip")
        return 0.0
    user_pref = user_row['preference'].iloc[0]

    # 2. 전체 메뉴 후보군 생성
    candidate_df = menu_df.copy()
    
    # 3. Feature 생성
    # A. CB Score
    candidate_df['CB_Score'] = candidate_df['menu_id'].apply(
        lambda x: cb.get_single_cb_score(x, user_pref)
    )
    
    # B. CF Score
    cf_scores = cf.get_predicted_scores(user_id, candidate_df['menu_id'].tolist())
    cf_score_map = {mid: score for mid, score in cf_scores}
    candidate_df['CF_Score'] = candidate_df['menu_id'].map(cf_score_map)
    
    # C. Graph Score (KeyError 방지를 위해 0.0으로 미리 초기화)
    candidate_df['Graph_Score'] = 0.0
    if not IS_BASELINE and gnn:
        candidate_df['Graph_Score'] = candidate_df['menu_id'].apply(
            lambda x: gnn.get_graph_score(user_id, x)
        )

        g_min = candidate_df['Graph_Score'].min()
        g_max = candidate_df['Graph_Score'].max()
        if g_max > g_min:
             candidate_df['Graph_Score'] = (candidate_df['Graph_Score'] - g_min) / (g_max - g_min)
    
    # D. Meta Scores
    candidate_df = pd.merge(candidate_df, rest_df[['rest_id', 'Latitude', 'Longitude', 'rating']], on='rest_id', how='left')
    candidate_df.rename(columns={'rating': 'Avg_Rating'}, inplace=True)
    candidate_df['Avg_Rating'] = candidate_df['Avg_Rating'].fillna(3.0)

    # 거리 점수: 후문('b') 기준으로 고정
    TARGET_LOCATION = 'b' 
    candidate_df['Distance_Score'] = candidate_df.apply(
        lambda row: calculate_distance_score(TARGET_LOCATION, row['Latitude'], row['Longitude']), 
        axis=1
    )
    
    # 4. 예측 (진단 모드 vs 일반 모드)
    if TEST_PURE_GNN and not IS_BASELINE:
        # [진단용] GNN 점수만으로 랭킹 산정 (MLP 무시)
        candidate_df['Predicted_Rating'] = candidate_df['Graph_Score']
    else:
        # [일반용] MLP 사용
        if IS_BASELINE:
            X_eval = candidate_df[['CB_Score', 'CF_Score', 'price', 'Distance_Score', 'Avg_Rating']].values
        else:
            X_eval = candidate_df[['CB_Score', 'CF_Score', 'Graph_Score', 'price', 'Distance_Score', 'Avg_Rating']].values
        
        candidate_df['Predicted_Rating'] = mlp.predict(X_eval)
    
    # 5. 결과 확인
    top_n_df = candidate_df.sort_values(by='Predicted_Rating', ascending=False).head(top_k)
    recommended_ids = top_n_df['menu_id'].tolist()
    
    # 6. Recall 계산
    hits = set(true_menu_ids) & set(recommended_ids)
    recall = len(hits) / len(true_menu_ids) if len(true_menu_ids) > 0 else 0.0
    
    if verbose:
        print(f"User {user_id}: 정답 {len(true_menu_ids)}개 중 {len(hits)}개 적중 -> Recall: {recall*100:.1f}%")
        
    return recall

def main():
    print(f"\n--- 전체 성능 평가 시작 (MODE: {'BASELINE' if IS_BASELINE else 'GNN'}) ---")
    if TEST_PURE_GNN and not IS_BASELINE:
        print("📢 [진단 모드] GNN 점수 단독 평가 중입니다.")

    # 1. 모델 로드
    models_data = load_models()
    if models_data[-1] is None: return
    
    ratings_df = models_data[2]

    # 2. 정답지(Ground Truth) 생성 (평점 4.0 이상만 정답으로 간주)
    good_ratings = ratings_df[ratings_df['rating'] >= 4.0]
    user_ground_truth = good_ratings.groupby('user_id')['menu_id'].apply(list).to_dict()
    
    print(f"\n총 {len(user_ground_truth)}명의 사용자에 대해 평가를 진행합니다.\n")

    # 3. 전체 사용자 평가
    total_recall = 0
    count = 0
    top_k = 10
    
    for user_id, true_ids in user_ground_truth.items():
        recall = evaluate_single_user(user_id, true_ids, models_data, top_k=top_k, verbose=True)
        total_recall += recall
        count += 1
        
    # 4. 최종 결과 출력
    avg_recall = total_recall / count if count > 0 else 0
    print("-" * 50)
    print(f"📊 최종 평가 결과 (Top-{top_k})")
    print(f"   - 총 평가 유저 수: {count}명")
    print(f"   - Average Recall : {avg_recall * 100:.2f}%")
    print("-" * 50)

if __name__ == "__main__":
    main()