import pandas as pd
import os

# 데이터 파일 경로 정의 (menu_data.csv)
DATA_PATH = '../data/menu_data.csv'

def clean_menu_data(file_path):
    """
    menu_data.csv 파일을 읽어 'price' 컬럼과 'features' 컬럼을 정제 후 저장합니다.
    - price: 쉼표 제거 및 정수 변환
    - features: 큰따옴표(")와 쉼표(,) 모두 제거하고 공백으로 치환
    """
    if not os.path.exists(file_path):
        print(f"Error: 파일을 찾을 수 없습니다: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error: 파일 로드 중 오류 발생 - {e}")
        return
        
    print(f"데이터 로드 완료. 총 {len(df)}개 레코드 처리 시작.")

    # --- 1. 'price' 컬럼 정제 (기존 로직 유지) ---
    def clean_price(price):
        if isinstance(price, str):
            try:
                # 쉼표 제거, 공백 제거 후 정수 변환
                return int(price.replace(',', '').strip())
            except ValueError:
                return None
        return price

    df.loc[:, 'price'] = df['price'].apply(clean_price)
    df['price'] = pd.to_numeric(df['price'], errors='coerce').astype('Int64') 


    # --- 2. 'features' 컬럼 정제 (새로운 로직) ---
    def clean_features(features):
        if isinstance(features, str):
            # 1. 큰따옴표(") 제거 (pandas가 파싱 시 제거하지 못한 경우 대비)
            features = features.replace('"', '').strip()
            
            # 2. 모든 쉼표(,)를 공백으로 치환하여 단어(토큰)를 분리
            features = features.replace(',', ' ')
            
            # 3. 중복 공백 제거 및 정리 (필요시)
            return ' '.join(features.split())
        return "" # 문자열이 아니거나 NaN인 경우 빈 문자열 반환

    df.loc[:, 'features'] = df['features'].apply(clean_features)


    # 3. 정제된 데이터를 원래 파일에 덮어쓰기 저장
    df.to_csv(file_path, index=False)
    
    print(f"✅ 파일 '{file_path}'의 'price'와 'features' 컬럼 정제 및 저장 완료.")
    print("이제 'features' 컬럼은 쉼표 없는 클린한 토큰 문자열입니다.")

if __name__ == '__main__':
    clean_menu_data(DATA_PATH)