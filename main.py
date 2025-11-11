# main.py

import pandas as pd
from filtering.contents_based import ContentBasedRecommender

# Pandas 출력 옵션 설정 (결과를 보기 편하게)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

def main():
    """
    메인 실행 함수
    """
    
    # --- (가정) 데이터 수집 단계 ---
    # data/menu_data.csv 파일이 이미 준비되었다고 가정
    # (예시 데이터 구조)
    # id, title,       price, location, category, features
    # 1,  김치찌개,     8000,  정문,     한식,     한식 찌개 돼지고기 김치 매운맛 8000원대
    # 2,  순두부찌개,   9000,  정문,     한식,     한식 찌개 해산물 순두부 매운맛 9000원대
    # 3,  간장새우덮밥, 10000, 후문,     일식,     일식 덮밥 간장 새우 10000원대
    # 4,  돈까스,       9500,  정문,     일식,     일식 튀김 돈까스 9000원대

    
    # 1. 사용자 정보 (하드코딩)
    # (향후 이 부분은 자연어 요청 분석 결과로 대체될 수 있음)
    user_info = {
        'user_id': '2020312857',
        'profile_text': '한식 찌개 얼얼한 매운맛 밥이랑', # 사용자의 선호 키워드
        'location': '북문',                           # 사용자의 현재 위치
        'max_price': 10000,                         # 사용자의 최대 예산
        'allergies': ['새우']                        # 사용자의 알레르기/불호 키워드
    }

    print(f"--- 사용자 요청 정보 ---")
    print(f"선호 키워드: {user_info['profile_text']}")
    print(f"위치: {user_info['location']}, 예산: {user_info['max_price']}원 이하, 제외: {user_info['allergies']}")
    print("-" * 30)

    # 2. 콘텐츠 기반 필터링 객체 생성
    # 객체 생성 시 data/menu_data.csv를 로드하고 TF-IDF를 학습 - 일단 dummy.csv
    try:
        recommender = ContentBasedRecommender(data_path='./data/dummy.csv')
    except Exception as e:
        print(f"추천기 초기화 중 오류 발생: {e}")
        return

    # 3. 추천 결과 요청
    recommendations = recommender.get_recommendations(user_info, top_n=5)
    
    # 4. 추천 결과 출력
    if not recommendations.empty:
        print(f"\n--- '정문' 근처 10000원 이하 '새우' 제외, '{user_info['profile_text']}' 키워드 기반 추천 결과 ---")
        print(recommendations)
    else:
        print("\n--- 아쉽지만, 모든 조건을 만족하는 추천 메뉴를 찾지 못했습니다. ---")
        print("필터 조건을 변경해보세요 (예: 예산 상향, 위치 변경)")


if __name__ == "__main__":
    main()