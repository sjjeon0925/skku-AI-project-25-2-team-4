# filtering/blender_mlp.py

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

class MLPBlender:
    def __init__(self, input_dim):
        # 1. MLP 모델 구조 정의 (Keras Sequential API 사용)
        self.scaler = StandardScaler()
        self.model = Sequential([
            # Input Layer (5개의 특징: CB_Score, CF_Score, Distance, Price, Rating)
            Dense(32, activation='relu', input_shape=(input_dim,)), 
            Dropout(0.2), # 과적합 방지를 위한 드롭아웃
            Dense(16, activation='relu'),
            # Output Layer: 예상 평점(1~5점)을 예측하는 회귀 문제이므로 활성화 함수는 'linear'
            Dense(1, activation='linear') 
        ])
        
        # 2. 모델 컴파일 (Loss: MSE, Metric: RMSE)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=[self.root_mean_squared_error] # RMSE를 직접 정의하여 사용
        )
        
    def root_mean_squared_error(self, y_true, y_pred):
        # RMSE는 MSE의 제곱근
        return np.sqrt(self.model.loss(y_true, y_pred))

    def train(self, X_train_raw, y_train, epochs=30, batch_size=32):
        """
        입력 특징(X)과 정답 평점(Y)을 받아 모델을 학습시킵니다.
        """
        # 특징 정규화 (Standardization)
        X_train = self.scaler.fit_transform(X_train_raw)
        
        # 훈련 및 검증 데이터셋 분리
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        # 모델 학습
        history = self.model.fit(
            X_train_split, y_train_split,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_split, y_val_split),
            verbose=1 # 학습 과정을 출력
        )
        return history

    def predict(self, X_predict_raw):
        """
        새로운 특징 벡터를 받아 최종 예상 평점을 예측합니다.
        """
        # 학습 시 사용한 Scaler로 정규화 (필수)
        X_predict = self.scaler.transform(X_predict_raw)
        return self.model.predict(X_predict).flatten()

# (참고: 이 코드를 사용하려면 TensorFlow를 설치해야 합니다: pip install tensorflow)