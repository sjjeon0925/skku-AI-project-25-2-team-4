# main.py

import pandas as pd
from filtering.contents_based import ContentBasedRecommender
from filtering.collaborative import CollaborativeRecommender

# Pandas 출력 옵션 설정 (결과를 보기 편하게)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# --- 테스트 모드 플래그 설정 ---
# 1: 콘텐츠 기반(CB) 단독
# 2: 협업 필터링(CF) 단독
# 3: 하이브리드 (CB + CF 조합)
RECOMMENDER_MODE = 3
# -----------------------------------


def load_menu_data(data_path='./data/menu_data.csv'):
    """메뉴 데이터를 로드하여 ID를 이름으로 매핑하는 딕셔너리를 반환"""
    try:
        menu_df = pd.read_csv(data_path)
        return dict(zip(menu_df['id'], menu_df['title']))
    except FileNotFoundError:
        print(f"메뉴 데이터 파일을 찾을 수 없습니다: {data_path}")
        return {}


def main():
    """메인 실행 함수: 플래그에 따라 추천 모드를 선택하여 실행"""
    
    # 1. 공통 데이터 및 추천기 초기화
    menu_id_to_title = load_menu_data()
    
    # --- 사용자 정보 (하드코딩 및 테스트용) ---
    # 협업 필터링은 user_id를 정수로 사용해야 함 (surprise 라이브러리 제약)
    TEST_USER_ID = 2020312857 
    
    user_info = {
        'user_id': TEST_USER_ID,
        'profile_text': '한식 찌개 얼큰한 매운맛 밥이랑', # CB용 선호 키워드
        'location': '정문',                           # 하드 필터: 위치
        'max_price': 10000,                         # 하드 필터: 예산
        'allergies': ['새우']                        # 하드 필터: 불호/알레르기
    }
    
    print("-" * 50)
    print(f"테스트 사용자 ID: {TEST_USER_ID}")
    
    if RECOMMENDER_MODE == 1:
        print("모드: 콘텐츠 기반(CB) 단독 실행")
    elif RECOMMENDER_MODE == 2:
        print("모드: 협업 필터링(CF) 단독 실행")
    elif RECOMMENDER_MODE == 3:
        print("모드: 하이브리드(CB + CF) 실행")
    else:
        print("유효하지 않은 RECOMMENDER_MODE 설정입니다.")
        return
    print("-" * 50)


    # 2. 추천기 인스턴스 생성 (모든 경우에 일단 생성)
    cb_recommender = ContentBasedRecommender()
    cf_recommender = CollaborativeRecommender()


    # 3. 추천 로직 실행 (모드별 분기)
    
    final_recommendations = []

    if RECOMMENDER_MODE == 1:
        # --- 모드 1: CB 단독 ---
        # CB의 하드 필터와 콘텐츠 유사도를 모두 사용하여 최종 결과를 얻음
        cb_results = cb_recommender.get_recommendations(user_info, top_n=10)
        final_recommendations = cb_results.rename(columns={'similarity_score': 'CB_Score'})

    elif RECOMMENDER_MODE == 2:
        # --- 모드 2: CF 단독 ---
        # CF 모델이 평가하지 않은 전체 메뉴를 대상으로 예상 별점을 계산함
        cf_results_raw = cf_recommender.get_top_n_recommendations(user_info['user_id'], top_n=10)
        
        # 결과를 DataFrame으로 변환하고 메뉴명 추가
        final_recommendations = pd.DataFrame(cf_results_raw, columns=['id', 'Predicted_Rating'])
        final_recommendations['title'] = final_recommendations['id'].map(menu_id_to_title)
        
    elif RECOMMENDER_MODE == 3:
        # --- 모드 3: 하이브리드 (CB 필터링 + CF 재정렬) ---
        
        # 3-1. [CB] 1차 후보군 필터링 (Top 20 메뉴 선택)
        # 하드 필터링과 콘텐츠 유사도를 사용하여 1차 후보군을 선정
        candidate_df = cb_recommender.get_recommendations(user_info, top_n=20)
        candidate_ids = candidate_df['id'].tolist()
        
        # 3-2. [CF] 2차 재정렬 (예상 별점 예측)
        # 후보군 메뉴에 대해 CF 모델의 예상 별점을 예측
        cf_re_rank_raw = cf_recommender.get_predicted_scores(user_info['user_id'], candidate_ids)
        
        # 3-3. 결과 병합 및 최종 순위 결정
        re_rank_df = pd.DataFrame(cf_re_rank_raw, columns=['id', 'Predicted_Rating'])
        
        # 원본 CB 결과와 CF 예상 별점 병합
        hybrid_df = pd.merge(candidate_df, re_rank_df, on='id', how='left')
        
        # Final Score: CB 유사도와 CF 예상 평점을 조합하여 최종 점수 산출 (예: 단순 곱)
        # 이 가중치는 프로젝트 목표에 맞게 조정 가능 (예: 0.6*CB + 0.4*CF)
        hybrid_df['Final_Hybrid_Score'] = (hybrid_df['similarity_score'] * 0.7) + (hybrid_df['Predicted_Rating'] / 5 * 0.3)
        
        # 최종 하이브리드 점수 기준으로 정렬 후 Top 10 선정
        final_recommendations = hybrid_df.sort_values(by='Final_Hybrid_Score', ascending=False).head(10)
        
    # 4. 최종 결과 출력
    if not final_recommendations.empty:
        print(f"\n최종 추천 결과 (Top {len(final_recommendations)})")
        print(final_recommendations)
    else:
        print("\n모든 조건을 만족하는 추천 메뉴를 찾지 못했습니다.")


if __name__ == "__main__":
    main()