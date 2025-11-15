import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class ContentBasedRecommender:
    """
    콘텐츠 기반 필터링 추천 시스템 클래스.
    메뉴의 특징(features)과 사용자의 선호(profile)를 기반으로 유사도를 계산합니다.
    """
    
    def __init__(self, data_path='./data/menu_data.csv'):
        """
        데이터를 로드하고 TF-IDF 행렬을 미리 학습시킵니다.
        """
        try:
            self.menu_df = pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"데이터 파일을 찾을 수 없습니다: {data_path}")
            # (실제 구현 시) 기본 데이터프레임 생성 또는 예외 처리
            self.menu_df = pd.DataFrame(columns=['id', 'title', 'features', 'price', 'location'])

        # 1. 데이터 전처리: 'features' 컬럼의 NaN 값을 빈 문자열로 대체
        self.menu_df['features'] = self.menu_df['features'].fillna('')
        
        # 2. TF-IDF 벡터화 객체 생성 및 학습
        # 메뉴의 모든 특징(키워드)을 학습
        self.tfidf_vectorizer = TfidfVectorizer()
        self.menu_feature_matrix = self.tfidf_vectorizer.fit_transform(self.menu_df['features'])
        
        print(f"콘텐츠 기반 모델 초기화 완료. 총 {self.menu_df.shape[0]}개 메뉴 로드됨.")

    def _apply_hard_filters(self, user_info):
        """
        (내부 함수) 사용자의 명시적 제약 조건(위치, 예산, 알레르기)을 적용하여
        추천 대상이 될 수 있는 메뉴의 인덱스(index)를 필터링합니다.
        """
        
        # 1. 원본 데이터프레임 복사
        filtered_df = self.menu_df.copy()
        
        # 2. 위치 필터링 (user_info에 'location'이 있는 경우)
        # location은 추후에 거리 정보를 수치화하여 유사도 계산 점수와 종합하는 버전으로 변경
        if user_info.get('location'):
            # 'location' 컬럼이 user_info의 위치와 일치하는 메뉴만 선택
            filtered_df = filtered_df[filtered_df['location'] == user_info['location']]
            
        # 3. 예산 필터링 (user_info에 'max_price'가 있는 경우)
        if user_info.get('max_price'):
            # 'price' 컬럼이 user_info의 최대 예산 이하인 메뉴만 선택
            filtered_df = filtered_df[filtered_df['price'] <= user_info['max_price']]
        
        # 4. 알레르기/불호 필터링 (user_info에 'allergies'가 있는 경우)
        if user_info.get('allergies'):
            # 'allergies'는 리스트라고 가정 (예: ['새우', '땅콩'])
            # 'features'에 알레르기 키워드가 포함되지 않은 메뉴만 선택
            pattern = '|'.join(user_info['allergies'])
            filtered_df = filtered_df[~filtered_df['features'].str.contains(pattern, na=False)]
            
        # 필터링된 메뉴의 원본 인덱스를 반환
        return filtered_df.index

    def get_recommendations(self, user_info, top_n=10):
        """
        사용자 정보를 받아 콘텐츠 기반 추천 메뉴를 반환합니다.

        :param user_info: (dict) 사용자 정보 딕셔너리
               (예: {'profile_text': '한식 찌개 얼큰한', 'location': '정문', 'max_price': 10000, 'allergies': ['새우']})
        :param top_n: (int) 추천할 메뉴 개수
        :return: (DataFrame) 추천 메뉴 상위 N개
        """
        
        # 1. 제약 조건(하드 필터)을 먼저 적용
        # 필터링을 통과한 메뉴들의 원본 인덱스 목록
        valid_indices = self._apply_hard_filters(user_info)
        
        if len(valid_indices) == 0:
            return pd.DataFrame(columns=['id', 'title', 'similarity_score', 'price', 'location'])

        # 2. 사용자 프로필(선호 키워드) 벡터화
        # user_info의 'profile_text' (예: "한식 고기 든든한")를 벡터로 변환
        user_profile_vector = self.tfidf_vectorizer.transform([user_info['profile_text']])
        
        # 3. 유사도 계산
        # 필터링된 메뉴들의 특징 행렬만 추출
        filtered_menu_matrix = self.menu_feature_matrix[valid_indices]
        
        # 사용자 프로필 벡터와 필터링된 메뉴 행렬 간의 코사인 유사도 계산
        similarity_scores = cosine_similarity(user_profile_vector, filtered_menu_matrix).flatten()
        
        # 4. 유사도 점수를 DataFrame에 추가
        # valid_indices를 인덱스로 사용하여 필터링된 메뉴 DataFrame을 가져옴
        recommendations_df = self.menu_df.loc[valid_indices].copy()
        recommendations_df['similarity_score'] = similarity_scores
        
        # 5. 유사도 점수 기준으로 내림차순 정렬하여 상위 N개 반환
        recommendations_df = recommendations_df.sort_values(by='similarity_score', ascending=False)
        
        return recommendations_df.head(top_n)[['id', 'title', 'similarity_score', 'price', 'location', 'features']]