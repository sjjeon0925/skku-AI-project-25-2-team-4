import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib

from filtering.contents_based import ContentBasedRecommender
from filtering.collaborative import CollaborativeRecommender
from filtering.blender_mlp import MLPBlender
from filtering.graph_model import GraphRecommender

from utils import (
    DATA_PATHS, calculate_distance_score, MODEL_DIR
)

def get_unrated_menu_ids(user_id, all_menu_ids, ratings_df):
    """사용자가 아직 평가하지 않은 메뉴 ID만 추출"""
    rated_menus = ratings_df[ratings_df['user_id'] == user_id]['menu_id'].values
    unrated_menus = [mid for mid in all_menu_ids if mid not in rated_menus]
    return unrated_menus

def get_cb_preference(user_id, user_df, query_str=""):
    """사용자 선호도 텍스트 + 쿼리 조합"""
    user_row = user_df[user_df['user_id'] == user_id]
    if user_row.empty:
        return query_str
    user_pref = user_row['preference'].iloc[0]
    if query_str:
        return f"{user_pref} {query_str}"
    return user_pref

def generate_prediction_features(candidate_df, user_id, user_loc, user_pref_full, 
                                 cb_recommender, cf_recommender, gnn_recommender, mode):
    
    X_predict_data = candidate_df.copy()

    # 1. CB Score
    # (이미 학습된 TF-IDF에 transform만 수행)
    X_predict_data['CB_Score'] = X_predict_data['menu_id'].apply(
        lambda menu_id: cb_recommender.get_single_cb_score(menu_id, user_pref_full)
    )

    # 2. CF Score
    cf_scores = cf_recommender.get_predicted_scores(user_id, X_predict_data['menu_id'].tolist())
    cf_score_map = {mid: score for mid, score in cf_scores}
    X_predict_data['CF_Score'] = X_predict_data['menu_id'].map(cf_score_map).fillna(0)

    # 3. Graph Score (Proposed / GNN Only 모드일 때만)
    X_predict_data['Graph_Score'] = 0.0
    if mode in ['proposed', 'gnn_only'] and gnn_recommender:
        X_predict_data['Graph_Score'] = X_predict_data['menu_id'].apply(
            lambda menu_id: gnn_recommender.get_graph_score(user_id, menu_id)
        )
        # Normalization (Min-Max in Candidate Set)
        g_min = X_predict_data['Graph_Score'].min()
        g_max = X_predict_data['Graph_Score'].max()
        if g_max > g_min:
             X_predict_data['Graph_Score'] = (X_predict_data['Graph_Score'] - g_min) / (g_max - g_min)

    # 4. Meta Scores
    X_predict_data['Distance_Score'] = X_predict_data.apply(
        lambda row: calculate_distance_score(user_loc, row['Latitude'], row['Longitude']), axis=1
    )
    
    # Rating 채우기 (결측치는 3.0)
    X_predict_data.rename(columns={'rating': 'Avg_Rating'}, inplace=True)
    X_predict_data['Avg_Rating'] = X_predict_data['Avg_Rating'].fillna(3.0)

    # 5. Feature Selection (Mode에 따라 다름)
    if mode == 'baseline':
        # [CB, CF, Price, Dist, Rating] -> Dim 5
        X = X_predict_data[['CB_Score', 'CF_Score', 'price', 'Distance_Score', 'Avg_Rating']].values
    elif mode == 'proposed':
        # [CB, CF, GNN, Price, Dist, Rating] -> Dim 6
        X = X_predict_data[['CB_Score', 'CF_Score', 'Graph_Score', 'price', 'Distance_Score', 'Avg_Rating']].values
    else: # gnn_only
        # MLP를 안 쓰지만 구조상 맞춤 (실제로는 GNN Score로만 정렬)
        X = X_predict_data[['Graph_Score']].values
    
    return X, X_predict_data

def main():
    parser = argparse.ArgumentParser(description="Menu Recommendation Prediction")
    parser.add_argument("--i", type=int, required=True, help="User ID")
    parser.add_argument("--l", type=str, required=False, default="b", help="Location Code (s, b, n, f)")
    parser.add_argument("--b", type=int, required=False, default=100000, help="Budget (KRW)")
    parser.add_argument("--q", type=str, required=False, default="", help="Additional Query")
    
    # 학습된 모델 정보 연동
    parser.add_argument('--mode', type=str, default='proposed', choices=['baseline', 'proposed', 'gnn_only'])
    parser.add_argument('--model_name', type=str, default='best_proposed')

    args = parser.parse_args()
    USER_ID, USER_LOC, USER_BUDGET, USER_QUERY = args.i, args.l, args.b, args.q
    
    print(f"\n🚀 [Prediction] User: {USER_ID} | Loc: {USER_LOC} | Mode: {args.mode}")

    # 1. 데이터 로드
    menu_df = pd.read_csv(DATA_PATHS['menu'])
    rest_df = pd.read_csv(DATA_PATHS['rest'])
    ratings_df = pd.read_csv(DATA_PATHS['rating'])
    user_df = pd.read_csv(DATA_PATHS['user'])
    
    # 메뉴 + 식당 정보 병합
    menu_df = pd.merge(menu_df, rest_df[['rest_id', 'rest_name', 'Latitude', 'Longitude', 'rating']], on='rest_id', how='left')

    # 2. 모델 로드
    # (1) CB / CF는 항상 로드 (Feature 생성용)
    cb_recommender = ContentBasedRecommender(DATA_PATHS['menu'])
    cf_recommender = CollaborativeRecommender(DATA_PATHS['rating'], DATA_PATHS['menu'])
    
    # (2) GNN 로드 (Proposed or GNN Only)
    gnn_recommender = None
    if args.mode in ['proposed', 'gnn_only']:
        gnn_path = os.path.join(MODEL_DIR, f"{args.model_name}_gnn.pth")
        if os.path.exists(gnn_path):
            gnn_recommender = GraphRecommender(DATA_PATHS['rating'], DATA_PATHS['menu'])
            gnn_recommender.load_model(gnn_path)
            print(f"✅ GNN Model Loaded: {gnn_path}")
        else:
            print(f"⚠️ GNN Model not found at {gnn_path}")

    # (3) MLP 로드 (Baseline or Proposed)
    mlp_blender = None
    if args.mode in ['baseline', 'proposed']:
        mlp_path = os.path.join(MODEL_DIR, f"{args.model_name}_mlp.keras")
        scaler_path = os.path.join(MODEL_DIR, f"{args.model_name}_scaler.joblib")
        
        if os.path.exists(mlp_path) and os.path.exists(scaler_path):
            input_dim = 5 if args.mode == 'baseline' else 6
            mlp_blender = MLPBlender(input_dim=input_dim)
            mlp_blender.model = tf.keras.models.load_model(mlp_path)
            mlp_blender.load_scaler(scaler_path)
            print(f"✅ MLP Model Loaded: {mlp_path}")
        else:
            print(f"❌ MLP Model not found! ({mlp_path})")
            return

    # 3. Candidate Generation (Hard Filtering)
    user_row = user_df[user_df['user_id'] == USER_ID]
    if user_row.empty:
        print(f"❌ User ID {USER_ID} not found.")
        return
        
    user_allergy = user_row['allergy'].iloc[0]
    
    # 알러지 & 가격 필터링
    if pd.isna(user_allergy):
        candidate_df = menu_df[menu_df['price'] <= USER_BUDGET].copy()
    else:
        candidate_df = menu_df[
            (menu_df['price'] <= USER_BUDGET) & 
            (~menu_df['features'].str.contains(user_allergy, na=False))
        ].copy()
    
    # 이미 먹어본 메뉴 제외
    unrated_ids = get_unrated_menu_ids(USER_ID, menu_df['menu_id'], ratings_df)
    candidate_df = candidate_df[candidate_df['menu_id'].isin(unrated_ids)]

    if candidate_df.empty:
        print("⚠️ 조건에 맞는 추천 후보가 없습니다.")
        return

    # 4. Feature 생성 및 점수 예측
    user_pref_full = get_cb_preference(USER_ID, user_df, USER_QUERY)
    
    X_predict, result_df = generate_prediction_features(
        candidate_df, USER_ID, USER_LOC, user_pref_full, 
        cb_recommender, cf_recommender, gnn_recommender, args.mode
    )
    
    # 최종 점수 계산
    if args.mode == 'gnn_only':
        # GNN Score 그대로 사용
        result_df['Final_Score'] = result_df['Graph_Score']
    else:
        # MLP 통과
        predicted_ratings = mlp_blender.predict(X_predict)
        result_df['Final_Score'] = predicted_ratings
    
    # 5. 결과 출력 (Top 10)
    top_n = result_df.sort_values(by='Final_Score', ascending=False).head(10)
    
    # 출력 포맷팅 유틸리티
    import unicodedata
    def get_display_width(text):
        width = 0
        for char in str(text):
            if unicodedata.east_asian_width(char) in ['F', 'W', 'A']: width += 2
            else: width += 1
        return width

    def pad_text(text, width):
        text = str(text)
        curr_w = get_display_width(text)
        return text + " " * max(0, width - curr_w)

    print("\n[Top 10 Recommendations]")
    h_menu = pad_text("Menu", 30)
    h_rest = pad_text("Restaurant", 30)
    h_price = pad_text("Price", 10)
    print(f"{h_menu} | {h_rest} | {h_price} | Score")
    print("-" * 90)

    for _, row in top_n.iterrows():
        menu_str = pad_text(row['menu'], 30)
        rest_str = pad_text(row['rest_name'], 30)
        price_str = pad_text(f"{row['price']:,}", 10)
        score_str = f"{row['Final_Score']:.4f}"
        
        print(f"{menu_str} | {rest_str} | {price_str} | {score_str}")

if __name__ == "__main__":
    main()