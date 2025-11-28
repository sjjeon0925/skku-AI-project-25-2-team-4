import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import joblib

# from sklearn.metrics import mean_squared_error

from filtering.contents_based import ContentBasedRecommender
from filtering.collaborative import CollaborativeRecommender
from filtering.blender_mlp import MLPBlender
from filtering.graph_model import GraphRecommender

from tensorflow.keras.models import load_model
from utils import (
    DATA_PATHS, calculate_distance_score, get_cb_preference,
    IS_BASELINE, MLP_MODEL_PATH, SCALER_PATH, GRAPH_MODEL_PATH, INPUT_FEATURE_DIM
)
# from sklearn.metrics.pairwise import cosine_similarity

# # --- 상수 설정 ---
# INPUT_FEATURE_DIM = 6
# MODEL_PATH = 'model/mlp_model.keras'
# GRAPH_MODEL_PATH = 'model/gnn_model.pth'
# SCALER_PATH = 'model/scaler.joblib' 

# LOCATION_NAME = {
#     's': '성균관대역',
#     'b': '후문',
#     'n': '북문',
#     'f': '정문'
# }

def get_unrated_menu_ids(user_id, all_menu_ids, ratings_df):
    rated_menus = ratings_df[ratings_df['user_id'] == user_id]['menu_id'].values
    unrated_menus = [mid for mid in all_menu_ids if mid not in rated_menus]
    return unrated_menus


def generate_prediction_features_predict(candidate_df, user_id, user_loc, user_pref_full, cb_recommender, cf_recommender, gnn_recommender=None):
    
    X_predict_data = candidate_df.copy()

    # 1. CB Score
    X_predict_data['CB_Score'] = X_predict_data['menu_id'].apply(
        lambda menu_id: cb_recommender.get_single_cb_score(menu_id, user_pref_full)
    )

    # 2. CF Score
    cf_scores = cf_recommender.get_predicted_scores(user_id, X_predict_data['menu_id'].tolist())
    cf_score_map = {mid: score for mid, score in cf_scores}
    X_predict_data['CF_Score'] = X_predict_data['menu_id'].map(cf_score_map)

    # 3. Graph Score (조건부)
    if not IS_BASELINE and gnn_recommender:
        X_predict_data['Graph_Score'] = X_predict_data['menu_id'].apply(
            lambda menu_id: gnn_recommender.get_graph_score(user_id, menu_id)
        )

    # 4. Meta Scores
    X_predict_data['Distance_Score'] = X_predict_data.apply(
        lambda row: calculate_distance_score(user_loc, row['Latitude'], row['Longitude']), axis=1
    )
    X_predict_data.rename(columns={'rating': 'Avg_Rating'}, inplace=True)
    X_predict_data['Avg_Rating'] = X_predict_data['Avg_Rating'].fillna(3.0)

    # X 추출 (분기)
    if IS_BASELINE:
        X_predict = X_predict_data[['CB_Score', 'CF_Score', 'price', 'Distance_Score', 'Avg_Rating']].values
    else:
        X_predict = X_predict_data[['CB_Score', 'CF_Score', 'Graph_Score', 'price', 'Distance_Score', 'Avg_Rating']].values
    
    return X_predict, X_predict_data


def main():
    parser = argparse.ArgumentParser(description="SKKU AI Project Recommendation System")
    parser.add_argument("--i", type=int, required=True, help="User ID")
    parser.add_argument("--l", type=str, required=True, help="Location Code (s, b, n, f)")
    parser.add_argument("--b", type=int, required=True, help="Budget (KRW)")
    parser.add_argument("--q", type=str, required=False, default="", help="Additional Query")
    
    args = parser.parse_args()
    USER_ID, USER_LOC, USER_BUDGET, USER_QUERY = args.i, args.l, args.b, args.q

    print(f"--- 예측 시작 (MODE: {'BASELINE' if IS_BASELINE else 'GNN'}) ---")
    print(f"User: {USER_ID}, Loc: {USER_LOC}, Budget: {USER_BUDGET}, Query: '{USER_QUERY}'")

    # 데이터 로드
    menu_df = pd.read_csv(DATA_PATHS['menu'])
    rest_df = pd.read_csv(DATA_PATHS['rest'])
    ratings_df = pd.read_csv(DATA_PATHS['rating'])
    user_df = pd.read_csv(DATA_PATHS['user'])
    
    menu_df = pd.merge(menu_df, rest_df[['rest_id', 'rest_name', 'Latitude', 'Longitude', 'rating']], on='rest_id', how='left')

    # 모델 초기화
    cb_recommender = ContentBasedRecommender(DATA_PATHS['menu'])
    cf_recommender = CollaborativeRecommender(DATA_PATHS['rating'], DATA_PATHS['menu'])
    
    gnn_recommender = None
    if not IS_BASELINE:
        # GNN 모드일 때만 GNN 모델 로드
        gnn_recommender = GraphRecommender(DATA_PATHS['rating'], DATA_PATHS['menu'])
        if os.path.exists(GRAPH_MODEL_PATH):
            gnn_recommender.load_model(GRAPH_MODEL_PATH)
        else:
            print("[Warning] GNN 모델 파일이 없습니다. GNN Score는 0으로 처리될 수 있습니다.")

    # MLP 모델 로드
    mlp_blender = MLPBlender(input_dim=INPUT_FEATURE_DIM)
    
    if os.path.exists(MLP_MODEL_PATH) and os.path.exists(SCALER_PATH):
        mlp_blender.model = tf.keras.models.load_model(MLP_MODEL_PATH)
        mlp_blender.load_scaler(SCALER_PATH)
        print(f"MLP 모델 로드 완료: {MLP_MODEL_PATH}")
    else:
        print(f"Error: MLP 모델 파일({MLP_MODEL_PATH})이 없습니다. train.py를 실행하세요.")
        return

    # 1. CB Preference 생성
    user_pref_full = get_cb_preference(USER_ID, USER_QUERY)
    
    # 2. Hard Filtering
    user_row = user_df[user_df['user_id'] == USER_ID]
    if user_row.empty:
        print(f"Error: User ID {USER_ID} not found in user_data.")
        return
        
    user_allergy = user_row['allergy'].iloc[0]
    
    if pd.isna(user_allergy):
        candidate_df = menu_df[menu_df['price'] <= USER_BUDGET].copy()
    else:
        candidate_df = menu_df[
            (menu_df['price'] <= USER_BUDGET) & 
            (~menu_df['features'].str.contains(user_allergy, na=False))
        ].copy()
    
    unrated_menu_ids = get_unrated_menu_ids(USER_ID, menu_df['menu_id'], ratings_df)
    candidate_df = candidate_df[candidate_df['menu_id'].isin(unrated_menu_ids)]

    if candidate_df.empty:
        print("조건에 맞는 메뉴가 없습니다.")
        return

    # 3. Feature 생성 및 예측
    X_predict, result_df = generate_prediction_features_predict(
        candidate_df, USER_ID, USER_LOC, user_pref_full, 
        cb_recommender, cf_recommender, gnn_recommender
    )
    
    predicted_ratings = mlp_blender.predict(X_predict)
    result_df['Predicted_Rating'] = predicted_ratings
    
    # 4. 결과 출력
    top_n = result_df.sort_values(by='Predicted_Rating', ascending=False).head(10)
    
    # print("\n[Top 10 Recommendations]")
    # print(f"{'Menu':<20} | {'Restaurant':<15} | {'Price':<8} | {'Score':<5}")
    # print("-" * 60)
    # for _, row in top_n.iterrows():
    #     print(f"{row['menu']:<20} | {row['rest_name']:<15} | {row['price']:<8} | {row['Predicted_Rating']:.2f}")
    import unicodedata

    def get_display_width(text):
        """한글은 2칸, 그 외는 1칸으로 너비 계산"""
        width = 0
        for char in str(text):
            if unicodedata.east_asian_width(char) in ['F', 'W', 'A']:
                width += 2
            else:
                width += 1
        return width

    def pad_text(text, width):
        """화면 너비(width)에 맞춰 공백을 추가하는 함수"""
        text = str(text)
        current_width = get_display_width(text)
        padding_len = max(0, width - current_width)
        return text + " " * padding_len

    # [수정] 결과 출력 부분
    print("\n[Top 10 Recommendations]")
    
    # 헤더 출력
    header_menu = pad_text("Menu", 60)
    header_rest = pad_text("Restaurant", 40) # 식당 이름이 좀 길 수 있어서 늘림
    header_price = pad_text("Price", 10)
    print(f"{header_menu} | {header_rest} | {header_price} | Score")
    print("-" * 60)

    # 데이터 출력
    for _, row in top_n.iterrows():
        menu_str = pad_text(row['menu'], 60)
        rest_str = pad_text(row['rest_name'], 40)
        price_str = pad_text(f"{row['price']:,}", 10) # 가격에 콤마(,) 추가
        score_str = f"{row['Predicted_Rating']:.2f}"
        
        print(f"{menu_str} | {rest_str} | {price_str} | {score_str}")

if __name__ == "__main__":
    main()