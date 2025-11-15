# filtering/collaborative.py

import pandas as pd
from surprise import SVD, Reader, Dataset
# from surprise.model_selection import build_full_trainset

class CollaborativeRecommender:
    """
    협업 필터링 (SVD 기반) 추천 시스템 클래스.
    사용자-아이템 평점 데이터를 기반으로 예상 별점을 예측합니다.
    """

    def __init__(self, ratings_path='./data/ratings_data.csv', menu_path='./data/menu_data.csv'):
        """
        데이터를 로드하고 SVD 모델을 학습시킵니다.
        """
        print("협업 필터링 모델을 초기화하고 학습합니다...")
        
        try:
            self.ratings_df = pd.read_csv(ratings_path)
            menu_df = pd.read_csv(menu_path)
            self.all_menu_ids = set(menu_df['id'].unique())
        except FileNotFoundError as e:
            print(f"데이터 파일 로드 오류: {e}. 협업 필터링 모델을 초기화할 수 없습니다.")
            self.model = None
            return

        # 1. Surprise 라이브러리용 Reader 및 Dataset 생성
        # rating_scale은 (최소 평점, 최대 평점)
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(self.ratings_df[['user_id', 'menu_id', 'rating']], reader)
        
        # 2. 전체 데이터를 학습 데이터셋으로 빌드
        trainset = data.build_full_trainset()
        
        # 3. SVD 모델 정의 및 학습
        # n_factors: 잠재 요인의 수 (하이퍼파라미터)
        # n_epochs: 학습 반복 횟수
        self.model = SVD(n_factors=100, n_epochs=20, random_state=42, verbose=False)
        self.model.fit(trainset)
        
        print("SVD 모델 학습 완료.")

    def get_predicted_scores(self, user_id, candidate_menu_ids):
        """
        [하이브리드 추천용 메소드]
        주어진 user_id와 후보 메뉴 ID 리스트에 대해 예상 평점을 반환합니다.

        :param user_id: (int or str) 사용자 ID
        :param candidate_menu_ids: (list) 콘텐츠 기반으로 필터링된 메뉴 ID 리스트
        :return: (list of tuples) [(menu_id, predicted_rating), ...] (예상 평점순 정렬)
        """
        if self.model is None:
            print("모델이 학습되지 않았습니다.")
            # 모델이 실패하면 후보군을 그냥 반환 (안전 장치)
            return [(menu_id, 0) for menu_id in candidate_menu_ids]
            
        predictions = []
        for menu_id in candidate_menu_ids:
            # .predict()는 (uid, iid, true_r, est, details) 튜플 반환
            # true_r (실제 평점)은 모르므로 None
            predicted = self.model.predict(uid=user_id, iid=menu_id, r_ui=None)
            predictions.append((menu_id, predicted.est)) # est: 예상 평점
            
        # 예상 평점(predicted.est)을 기준으로 내림차순 정렬
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        return predictions

    def get_top_n_recommendations(self, user_id, top_n=10):
        """
        [협업 필터링 단독 테스트용 메소드]
        사용자가 평가하지 않은 모든 메뉴에 대해 예상 평점을 계산하고 상위 N개를 반환합니다.
        (참고: 전체 메뉴 대상이라 하이브리드 방식보다 느릴 수 있습니다.)

        :param user_id: (int or str) 사용자 ID
        :param top_n: (int) 추천할 개수
        :return: (list of tuples) [(menu_id, predicted_rating), ...] (상위 N개)
        """
        if self.model is None:
            print("모델이 학습되지 않았습니다.")
            return []
            
        # 1. 사용자가 이미 평가한 메뉴 ID 셋
        rated_menus = set(self.ratings_df[self.ratings_df['user_id'] == user_id]['menu_id'])
        
        # 2. 평가하지 않은 메뉴 ID 셋
        unrated_menus = self.all_menu_ids - rated_menus
        
        # 3. 평가하지 않은 메뉴들에 대해 예상 평점 계산
        # (get_predicted_scores 메소드 재활용)
        predictions = self.get_predicted_scores(user_id, list(unrated_menus))
        
        # 4. 상위 N개 반환
        return predictions[:top_n]