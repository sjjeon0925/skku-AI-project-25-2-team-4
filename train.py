import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from filtering.contents_based import ContentBasedRecommender
from filtering.collaborative import CollaborativeRecommender
from filtering.blender_mlp import MLPBlender
from utils import DATA_PATHS, calculate_distance_score # [수정] utils에서 상수 및 함수 임포트

# --- 학습 설정 ---
MODEL_PATH = 'model/final_mlp_model.keras'
SCALER_PATH = 'model/scaler.joblib' 
# INPUT_FEATURE_DIM = 5
EPOCHS = 300 
PLOT_FILENAME = 'model/training_rmse_plot.png'

def plot_training_history(history, filename):
    """Keras 학습 history 객체를 받아 Train/Validation RMSE 그래프를 출력합니다."""
    
    # history.history에서 RMSE 값을 추출
    
    # history.history에 'root_mean_squared_error'와 'val_root_mean_squared_error'가 있다고 가정
    train_rmse = history.history['root_mean_squared_error']
    val_rmse = history.history['val_root_mean_squared_error']
    epochs = range(1, len(train_rmse) + 1)
    
    plt.figure(figsize=(10, 6))
    
    # 훈련 RMSE (파란색)
    plt.plot(epochs, train_rmse, 'b', label='Training RMSE')
    # 검증 RMSE (빨간색)
    plt.plot(epochs, val_rmse, 'r', label='Validation RMSE')
    
    plt.title('Training and Validation RMSE by Epoch')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    
    # 저장 경로가 없으면 생성
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    plt.close() # 메모리 해제
    print(f"\n✅ Training plot saved to {filename}")

def generate_hybrid_features_train(ratings_df, menu_df, rest_df, user_df, cb_recommender, cf_recommender):
    """
    MLP 학습에 사용할 X (입력 특징)와 Y (정답 평점) 데이터를 생성합니다.
    (Train 시에는 USER_QUERY/USER_LOC_CHAR 대신 과거 기록을 사용)
    """
    print("\n[3] 하이브리드 특징 행렬 (X, Y) 생성 시작...")
    
    # 1. 데이터 병합 (rating_df -> menu_df -> rest_df)
    data = ratings_df.copy()
    data = pd.merge(data, menu_df[['menu_id', 'rest_id', 'price', 'features']], on='menu_id', how='left')
    data = pd.merge(data, rest_df[['rest_id', 'Latitude', 'Longitude', 'rating']], on='rest_id', how='left', suffixes=('_menu', '_rest')) 
    
    # 2. 특징 계산 (과거 기록 기반)
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
    
    # Distance Score: 평가 당시 위치를 사용
    # NOTE: location_map은 utils.py의 COORDINATES와 매칭되어야 합니다.
    location_map = {'성균관대역': 's', '정문': 'f', '후문': 'b', '북문': 'n'} 
    data['Distance_Score'] = data.apply(
        lambda row: calculate_distance_score(
            location_map.get(row['location'], 'f'), # 평가 당시 위치 (문자열->코드)
            row['Latitude'], row['Longitude']
        ),
        axis=1
    )
    
    # 3. X, Y 추출
    X = data[['CB_Score', 'CF_Score', 'price', 'Distance_Score', 'rating_rest']].values
    Y = data['rating_menu'].values 
    
    print(f"특징 행렬 X 생성 완료. Shape: {X.shape}")
    return X, Y


def main():
    # 1. 모든 데이터 로드
    try:
        ratings_df = pd.read_csv(DATA_PATHS['rating'])
        menu_df = pd.read_csv(DATA_PATHS['menu'])
        rest_df = pd.read_csv(DATA_PATHS['rest'])
        user_df = pd.read_csv(DATA_PATHS['user'])
    except FileNotFoundError as e:
        print(f"Error: 필수 데이터 파일 로드 실패: {e}")
        return

    # 2. 추천기 인스턴스 초기화 (CF 모델 학습 포함)
    cb_recommender = ContentBasedRecommender(data_path=DATA_PATHS['menu'])
    cf_recommender = CollaborativeRecommender(ratings_path=DATA_PATHS['rating'], menu_path=DATA_PATHS['menu'])
    
    # 3. 특징 행렬 X, Y 생성 (학습 데이터)
    X_train_full, Y_train_full = generate_hybrid_features_train(ratings_df, menu_df, rest_df, user_df, cb_recommender, cf_recommender)
    
    # 4. 학습/검증 셋 분리
    X_train, X_test, Y_train, Y_test = train_test_split(
        X_train_full, Y_train_full, test_size=0.2, random_state=42
    )

    # 5. MLP 모델 학습 및 저장
    print("\n[5] MLP 모델 학습 시작...")
    
    mlp_blender = MLPBlender(input_dim=X_train_full.shape[1])
    
    # 모델 학습
    history = mlp_blender.train(X_train, Y_train, epochs=EPOCHS, batch_size=4) 
    
    # 6. 학습 완료 후 시각화 및 모델/Scaler 저장
    
    # RMSE 변화 그래프 출력
    plot_training_history(history, PLOT_FILENAME) 
    
    # 모델 및 Scaler 저장
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    mlp_blender.model.save(MODEL_PATH)
    mlp_blender.save_scaler(SCALER_PATH) 
    
    print(f"\n✅ 학습 완료! 모델: {MODEL_PATH}, 스케일러: {SCALER_PATH} 저장됨.")
    
    # 7. 검증 데이터셋 성능 출력
    Y_pred_test = mlp_blender.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(Y_test, Y_pred_test))
    print(f"✅ 테스트 셋 RMSE: {test_rmse:.4f}")


if __name__ == "__main__":
    
    main()