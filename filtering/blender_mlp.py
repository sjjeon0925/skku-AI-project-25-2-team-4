from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import joblib

# 기본값 상수 (외부 인자가 없을 때 사용)
MLP_EPOCHS = 300
MLP_BATCH_SIZE = 4
MLP_LR = 0.001
LOG_INTERVAL = 10

class IntervalLogger(Callback):
    def __init__(self, interval):
        super(IntervalLogger, self).__init__()
        self.interval = interval

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            msg = f"Epoch {epoch + 1}: "
            for k, v in logs.items():
                msg += f"- {k}: {v:.4f} "
            print(msg)

class MLPBlender:
    def __init__(self, input_dim):
        self.scaler = StandardScaler()
        self.model = Sequential([
            Dense(32, activation='relu', input_shape=(input_dim,)), 
            Dropout(0.2), 
            Dense(16, activation='relu'),
            Dense(1, activation='linear') 
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=MLP_LR),
            loss='mse',
            metrics=[RootMeanSquaredError()]
        )

    # [수정] epochs, batch_size 인자 추가 및 기본값 설정
    def train(self, X_train_raw, y_train, epochs=MLP_EPOCHS, batch_size=MLP_BATCH_SIZE, log_interval=LOG_INTERVAL):
        """
        입력 특징(X)과 정답 평점(Y)을 받아 모델을 학습시킵니다.
        """
        X_train = self.scaler.fit_transform(X_train_raw)
        
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        history = self.model.fit(
            X_train_split, y_train_split,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val_split, y_val_split),
            verbose=0,
            callbacks=[IntervalLogger(log_interval)]
        )
        return history

    def predict(self, X_predict_raw):
        X_predict = self.scaler.transform(X_predict_raw)
        return self.model.predict(X_predict).flatten()

    def save_scaler(self, path):
        joblib.dump(self.scaler, path)

    def load_scaler(self, path):
        self.scaler = joblib.load(path)