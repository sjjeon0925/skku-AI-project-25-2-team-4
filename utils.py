import math
import pandas as pd
import os
from sklearn.metrics import mean_squared_error

# --- 🎯 위치 정보 및 상수 정의 ---
COORDINATES = {
    's': (37.29986776148395, 126.97219805873624), # 성균관대역
    'b': (37.29633029410662, 126.97061603024721), # 후문 (Back gate)
    'n': (37.296274335479666, 126.9764159771293), # 북문 (North gate)
    'f': (37.29100570424096, 126.97417156623229), # 정문 (Front gate)
}
R = 6371 # 지구 반지름 (km)

DATA_PATHS = {
    'menu': './data/menu_data.csv',
    'rest': './data/rest_data.csv',
    'user': './data/user_data.csv',
    'rating': './data/rating_data.csv',
}

# --- 지리 계산 함수 ---

def haversine(lat1, lon1, lat2, lon2):
    """하버사인 공식을 사용하여 두 좌표 간의 거리를 km 단위로 계산합니다."""
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def calculate_distance_score(user_loc_char, rest_lat, rest_lon):
    """현재 사용자 위치와 식당 좌표 간의 거리 점수를 계산합니다."""
    if user_loc_char not in COORDINATES: return 0.0

    user_lat, user_lon = COORDINATES[user_loc_char]
    distance_km = haversine(user_lat, user_lon, rest_lat, rest_lon)

    L0 = 0.5 # 특성 거리
    score = math.exp(-distance_km / L0)
    return score

def get_cb_preference(user_id, query_str=None):
    """
    CB Score 계산에 사용할 최종 선호도 문자열을 결정합니다.
    """
    user_df = pd.read_csv(DATA_PATHS['user'])
    user_pref = user_df[user_df['user_id'] == user_id]['preference'].iloc[0]
    
    if query_str is None or pd.isna(query_str) or query_str == "":
        return user_pref
    
    # 쿼리가 있으면 기존 선호도와 결합
    return user_pref + " " + query_str