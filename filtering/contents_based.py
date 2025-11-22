import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils import DATA_PATHS 

class ContentBasedRecommender:
    """
    콘텐츠 기반 필터링 추천 시스템 클래스.
    메뉴 특징(features)과 사용자의 선호(profile)를 기반으로 유사도를 계산합니다.
    """
    
    def __init__(self, data_path='./data/menu_data.csv'):
        # 1. 데이터 로드 및 전처리
        self.menu_df = pd.read_csv(data_path)
        self.menu_df['features'] = self.menu_df['features'].fillna('')
        
        # 2. 특징 벡터화 (TF-IDF)
        self.tfidf_vectorizer = TfidfVectorizer()
        self.menu_feature_matrix = self.tfidf_vectorizer.fit_transform(self.menu_df['features'])
        
        print(f"콘텐츠 기반 모델 초기화 완료. 총 {self.menu_df.shape[0]}개 메뉴 로드됨.")


    # Train과 Predict 로직 모두 처리 가능
    def get_single_cb_score(self, menu_id, user_pref_source, user_df=None):
        """
        특정 메뉴 ID와 사용자 선호도 정보를 받아 코사인 유사도 점수를 계산합니다.

        :param menu_id: 추천할 메뉴 ID
        :param user_pref_source: user_id (Train 시) 또는 최종 선호도 문자열 (Predict 시)
        :param user_df: Predict 시에는 None, Train 시에는 user_df 전체가 전달됨
        :return: 코사인 유사도 점수 (float)
        """
        
        # 1. 선호도 문자열 결정 (훈련 vs. 예측)
        if isinstance(user_pref_source, int):
            # A. 훈련 시나리오: user_id가 들어왔을 때 DB(user_df)에서 조회
            user_id = user_pref_source
            if user_df is None: 
                # user_df가 인수로 전달되지 않은 경우, 파일을 로드하여 조회 (안전 장치)
                try: user_df = pd.read_csv(DATA_PATHS['user'])
                except: return 0.0
                
            user_pref = user_df[user_df['user_id'] == user_id]['preference'].iloc[0]
        
        elif isinstance(user_pref_source, str):
            # B. 예측 시나리오: 쿼리가 결합된 최종 선호도 문자열이 직접 들어왔을 때
            user_pref = user_pref_source
            
        else:
            return 0.0

        # 2. TF-IDF 벡터 생성 및 유사도 계산
        # 'menu_id' 컬럼 이름이 'id'가 아닌 'menu_id'인지 확인
        menu_index = self.menu_df[self.menu_df['menu_id'] == menu_id].index
        if len(menu_index) == 0: return 0.0
        
        # 사용자 선호도 문자열을 TF-IDF 벡터로 변환
        user_vector = self.tfidf_vectorizer.transform([user_pref])
        
        # 메뉴 특징 벡터 추출
        menu_vector = self.menu_feature_matrix[menu_index[0]]
        
        # 코사인 유사도 계산
        return cosine_similarity(user_vector, menu_vector)[0][0]
