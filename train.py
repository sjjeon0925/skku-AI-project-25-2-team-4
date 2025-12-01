import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import torch

# [유지] 기존 임포트 경로 및 파일 구조 준수
from filtering.contents_based import ContentBasedRecommender
from filtering.collaborative import CollaborativeRecommender
from filtering.blender_mlp import MLPBlender
from filtering.graph_model import GraphRecommender

from utils import (
    DATA_PATHS, calculate_distance_score, IS_BASELINE, 
    MLP_MODEL_PATH, SCALER_PATH, GRAPH_MODEL_PATH
)

# [유지] PLOT_FILENAME 설정 로직
if IS_BASELINE:
    PLOT_FILENAME = 'model/training_rmse_plot_baseline.png'
else:
    PLOT_FILENAME = 'model/training_rmse_plot_gnn.png'

def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def generate_negative_samples(ratings_df, menu_df, ratio=1):
    """
    [추가] 안 가본 식당(평점 0) 데이터를 생성하여 MLP 학습용으로 반환
    """
    print(f"   >> 네거티브 샘플링 생성 중... (Positive 1 : Negative {ratio})")
    
    users = ratings_df['user_id'].unique()
    all_menus = menu_df['menu_id'].unique()
    # 검색 속도를 위해 Set으로 변환
    user_visited = ratings_df.groupby('user_id')['menu_id'].apply(set).to_dict()
    
    neg_rows = []
    for user_id in users:
        visited = user_visited.get(user_id, set())
        unvisited = list(set(all_menus) - visited)
        
        if not unvisited: continue
            
        # 방문한 개수만큼(ratio=1) 랜덤 추출
        num_neg = int(len(visited) * ratio)
        if num_neg == 0 and len(visited) > 0: num_neg = 1 # 최소 1개 보장
        
        selected_neg = random.sample(unvisited, min(num_neg, len(unvisited)))
        
        for menu_id in selected_neg:
            neg_rows.append({
                'user_id': user_id,
                'menu_id': menu_id,
                'rating_menu': 0.0, # 0점 부여 (중요)
            })
            
    print(f"   >> 네거티브 샘플 {len(neg_rows)}개 생성 완료.")
    return pd.DataFrame(neg_rows)

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
    
    # 1. Positive Data (ratings_df는 main에서 이미 rating_menu로 변경됨)
    pos_data = ratings_df.copy()
    
    # 2. Negative Data 추가 (GNN 모드일 때만 적용)
    if not IS_BASELINE: 
        neg_data = generate_negative_samples(pos_data, menu_df, ratio=1)
        data = pd.concat([pos_data, neg_data], ignore_index=True)
    else:
        data = pos_data

    # 3. 메뉴 정보 병합
    data = pd.merge(data, menu_df[['menu_id', 'rest_id', 'price', 'features']], on='menu_id', how='left')
    
    # 4. 식당 정보 병합 (rating -> rating_rest 이름 변경)
    rest_temp = rest_df[['rest_id', 'Latitude', 'Longitude', 'rating']].rename(columns={'rating': 'rating_rest'})
    data = pd.merge(data, rest_temp, on='rest_id', how='left')    
    
    # 5. 결측치 처리 (네거티브 샘플용)
    if not IS_BASELINE:
        data['price'] = data['price'].fillna(data['price'].mean())
        data['rating_rest'] = data['rating_rest'].fillna(3.0)
        # 위치 정보는 학습에 큰 영향 없으므로 기본값(후문) 처리
        data['Latitude'] = data['Latitude'].fillna(37.2963)
        data['Longitude'] = data['Longitude'].fillna(126.9706)

    # 6. 특징 계산
    print(" - CB Score 계산 중...")
    # apply 속도 최적화를 위해 dict 변환
    user_pref_dict = user_df.set_index('user_id')['preference'].to_dict()
    
    data['CB_Score'] = data.apply(
        lambda row: cb_recommender.get_single_cb_score(
            row['menu_id'], 
            user_pref_dict.get(row['user_id'], "")
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
        
        # [중요] Min-Max Scaling (0~1 정규화)
        min_score = data['Graph_Score'].min()
        max_score = data['Graph_Score'].max()
        if max_score > min_score:
            data['Graph_Score'] = (data['Graph_Score'] - min_score) / (max_score - min_score)
        
        print(f"   >> GNN Score Range: {min_score:.4f} ~ {max_score:.4f} (Normalized)")
    
    # Distance Score: 평가 당시 위치 사용 (네거티브 샘플은 'f' 정문 기준 등 임의 설정)
    location_map = {'성균관대역': 's', '정문': 'f', '후문': 'b', '북문': 'n'} 
    data['Distance_Score'] = data.apply(
        lambda row: calculate_distance_score(
            location_map.get(row.get('location', 'f'), 'f'), 
            row['Latitude'], row['Longitude']
        ),
        axis=1
    )

    # 7. X, Y 추출
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
        
        # [필수] 컬럼명 변경 (rating -> rating_menu)
        if 'rating' in ratings_df.columns:
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
        gnn_recommender.train() # 인자 없이 호출 (내부 상수 사용)
        
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
    # 네거티브 샘플링으로 데이터가 늘어났으므로 확인용 출력
    print(f"    - 총 학습 데이터 수: {len(Y_train_full)}")

    mlp_blender = MLPBlender(input_dim=X_train_full.shape[1]) 
    
    # train 함수는 blender_mlp.py의 상수를 사용하므로 인자 없이 데이터만 전달
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
    set_seeds(41)
    main()