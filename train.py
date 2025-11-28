import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error

from filtering.contents_based import ContentBasedRecommender
from filtering.collaborative import CollaborativeRecommender
from filtering.blender_mlp import MLPBlender
from filtering.graph_model import GraphRecommender

from utils import (
    DATA_PATHS, calculate_distance_score, IS_BASELINE, 
    MLP_MODEL_PATH, SCALER_PATH, GRAPH_MODEL_PATH
)

if IS_BASELINE:
    PLOT_FILENAME = 'model/training_rmse_plot_baseline.png'
else:
    PLOT_FILENAME = 'model/training_rmse_plot_gnn.png'

def plot_training_history(history):
    rmse = history.history['root_mean_squared_error']
    val_rmse = history.history['val_root_mean_squared_error']
    epochs = range(1, len(rmse) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, rmse, 'b-', label='Training RMSE')
    plt.plot(epochs, val_rmse, 'r-', label='Validation RMSE')
    plt.title('Training and Validation RMSE')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(PLOT_FILENAME), exist_ok=True)
    plt.savefig(PLOT_FILENAME)
    print(f"학습 그래프 저장 완료: {PLOT_FILENAME}")

def generate_hybrid_features_train(ratings_df, menu_df, rest_df, user_df, cb_recommender, cf_recommender, gnn_recommender=None):
    print("\n[3] 하이브리드 특징 행렬 (X, Y) 생성 시작...")
    
    # 1. 데이터 병합
    data = ratings_df.copy()
    data = pd.merge(data, menu_df[['menu_id', 'rest_id', 'price', 'features']], on='menu_id', how='left')
    rest_temp = rest_df[['rest_id', 'Latitude', 'Longitude', 'rating']].rename(columns={'rating': 'rating_rest'})
    data = pd.merge(data, rest_temp, on='rest_id', how='left')    
    
    # 2. 특징 계산 (과거 기록 기반)
    print(" - CB Score 계산 중...")
    data['CB_Score'] = data.apply(
        lambda row: cb_recommender.get_single_cb_score(
            row['menu_id'], 
            user_df[user_df['user_id'] == row['user_id']]['preference'].iloc[0]
        ), axis=1
    )
    
    print(" - CF Score 계산 중...")
    data['CF_Score'] = data.apply(
        lambda row: cf_recommender.model.predict(
            uid=row['user_id'], iid=row['menu_id']
        ).est,
        axis=1
    )

    if not IS_BASELINE and gnn_recommender:
        print(" - Graph Score 계산 중...")
        data['Graph_Score'] = data.apply(
            lambda row: gnn_recommender.get_graph_score(row['user_id'], row['menu_id']),
            axis=1
        ).fillna(0)
    
    # Distance Score: 평가 당시 위치를 사용
    location_map = {'성균관대역': 's', '정문': 'f', '후문': 'b', '북문': 'n'} 
    data['Distance_Score'] = data.apply(
        lambda row: calculate_distance_score(
            location_map.get(row['location'], 'f'), # 평가 당시 위치
            row['Latitude'], row['Longitude']
        ),
        axis=1
    )

    # 3. X, Y 추출
    if IS_BASELINE:
        # [Baseline] 5 Features
        X = data[['CB_Score', 'CF_Score', 'price', 'Distance_Score', 'rating_rest']].values
    else:
        # [GNN] 6 Features
        X = data[['CB_Score', 'CF_Score', 'Graph_Score', 'price', 'Distance_Score', 'rating_rest']].values
        
    Y = data['rating_menu'].values
    
    print(f"특징 행렬 X 생성 완료. Shape: {X.shape}")
    return X, Y


def main():
    print(f"--- 학습 시작 (MODE: {'BASELINE' if IS_BASELINE else 'GNN'}) ---")

    # 1. 모든 데이터 로드
    try:
        ratings_df = pd.read_csv(DATA_PATHS['rating'])
        menu_df = pd.read_csv(DATA_PATHS['menu'])
        rest_df = pd.read_csv(DATA_PATHS['rest'])
        user_df = pd.read_csv(DATA_PATHS['user'])
        ratings_df.rename(columns={'rating': 'rating_menu'}, inplace=True)
    except FileNotFoundError as e:
        print(f"Error: 필수 데이터 파일 로드 실패: {e}")
        return

    # 2. 추천기 인스턴스 초기화 (CF 모델 학습 포함)
    cb_recommender = ContentBasedRecommender(data_path=DATA_PATHS['menu'])
    cf_recommender = CollaborativeRecommender(ratings_path=DATA_PATHS['rating'], menu_path=DATA_PATHS['menu'])

    # 2-1. GNN 모델 초기화, 학습 및 저장
    gnn_recommender = None
    if not IS_BASELINE:
        print("GNN(LightGCN) 모델 학습...")
        gnn_recommender = GraphRecommender(ratings_path=DATA_PATHS['rating'], menu_path=DATA_PATHS['menu'])
        gnn_recommender.train()
        
        os.makedirs(os.path.dirname(GRAPH_MODEL_PATH), exist_ok=True)
        gnn_recommender.save_model(GRAPH_MODEL_PATH)
    else:
        print(">> IS_BASELINE=True: GNN 학습을 건너뜁니다.")
    
    # 3. 특징 행렬 X, Y 생성 (학습 데이터)
    X_train_full, Y_train_full = generate_hybrid_features_train(
        ratings_df, menu_df, rest_df, user_df, 
        cb_recommender, cf_recommender, gnn_recommender
    )
 
    # 4. MLP 모델 학습
    print(f"\n[5] MLP 모델 학습 시작... (Target: {MLP_MODEL_PATH})")
    mlp_blender = MLPBlender(input_dim=X_train_full.shape[1]) # 자동으로 5 또는 6 설정됨
    
    history = mlp_blender.train(
        X_train_full, 
        Y_train_full
    ) 
    
    # 5. 모델 저장
    os.makedirs(os.path.dirname(MLP_MODEL_PATH), exist_ok=True)
    mlp_blender.model.save(MLP_MODEL_PATH)
    mlp_blender.save_scaler(SCALER_PATH)
    print(f"최종 모델 및 스케일러 저장 완료.\n - Model: {MLP_MODEL_PATH}\n - Scaler: {SCALER_PATH}")
    
    plot_training_history(history)

if __name__ == "__main__":
    main()