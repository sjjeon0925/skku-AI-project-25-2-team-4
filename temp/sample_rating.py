import pandas as pd
import numpy as np
import os

# --- 상수 정의 ---
INPUT_FILE = './data/rating_data.csv'
OUTPUT_FILE = './data/sampled_ratings_data.csv'
MIN_RATINGS = 15
MAX_RATINGS = 20
SEED = 42 # 재현성을 위해 시드 고정

def create_sparse_ratings(input_path, output_path, min_n, max_n, seed):
    # 1. 데이터 로드
    if not os.path.exists(input_path):
        print(f"Error: 파일을 찾을 수 없습니다: {input_path}")
        return

    df = pd.read_csv(input_path)
    print(f"원본 평점 데이터 로드 완료. 총 {len(df)}개 평점.")

    # 2. 사용자별 샘플링 함수 정의
    def sample_ratings(group):
        # 각 유저 그룹에 대해 15~20개 사이의 랜덤 샘플 크기를 결정
        n_samples = np.random.randint(min_n, max_n + 1)
        
        # 실제 그룹 크기보다 n_samples가 클 수 없으므로, 그룹 크기로 제한
        n_samples = min(n_samples, len(group))
        
        return group.sample(n=n_samples, random_state=seed)

    # 3. 샘플링 적용
    # user_id를 기준으로 그룹화하고, 각 그룹에 대해 sample_ratings 함수 적용
    sampled_df = df.groupby('user_id', group_keys=False).apply(sample_ratings).reset_index(drop=True)

    # 4. 결과 저장
    # 디렉토리가 없으면 생성 (안전성 확보)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sampled_df.to_csv(output_path, index=False)
    
    # 5. 결과 보고
    total_ratings = len(sampled_df)
    unique_users = sampled_df['user_id'].nunique()
    
    print(f"✅ 희소화 완료! {unique_users}명의 유저에 대해 총 {total_ratings}개의 평점 데이터가 생성되었습니다.")
    print(f"새로운 데이터는 {output_path}에 저장되었습니다.")
    print("\n--- 샘플링 결과 확인 (유저당 평점 수) ---")
    print(sampled_df['user_id'].value_counts().head())

# 스크립트 실행
if __name__ == '__main__':
    create_sparse_ratings(INPUT_FILE, OUTPUT_FILE, MIN_RATINGS, MAX_RATINGS, SEED)