from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib

class MLPBlender:
    def __init__(self, input_dim):
        # 1. MLP 모델 구조 정의 (Keras Sequential API 사용)
        self.scaler = StandardScaler()
        self.model = Sequential([
            Dense(32, activation='relu', input_shape=(input_dim,)), 
            Dropout(0.2), 
            Dense(16, activation='relu'),
            Dense(1, activation='linear') 
        ])
        
        # 2. 모델 컴파일 (Loss: MSE, Metric: RMSE)
        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse',
            metrics=[RootMeanSquaredError()]
        )

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
        # 학습 시 사용한 Scaler로 정규화
        X_predict = self.scaler.transform(X_predict_raw)
        return self.model.predict(X_predict).flatten()

    # Scaler 저장/로드 기능
    def save_scaler(self, path):
        joblib.dump(self.scaler, path)

    def load_scaler(self, path):
        self.scaler = joblib.load(path)