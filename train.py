import argparse
import pandas as pd
import numpy as np
import os
import random
import tensorflow as tf
import torch
from sklearn.model_selection import KFold

# 모듈 임포트
from filtering.contents_based import ContentBasedRecommender
from filtering.collaborative import CollaborativeRecommender
from filtering.blender_mlp import MLPBlender
from filtering.graph_model import GraphRecommender
from utils import DATA_PATHS, calculate_distance_score, MODEL_DIR

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def generate_negative_samples(ratings_df, menu_df, ratio=1):
    users = ratings_df['user_id'].unique()
    all_menus = menu_df['menu_id'].unique()
    user_visited = ratings_df.groupby('user_id')['menu_id'].apply(set).to_dict()
    neg_rows = []
    for user_id in users:
        visited = user_visited.get(user_id, set())
        unvisited = list(set(all_menus) - visited)
        if not unvisited: continue
        num_neg = max(1, int(len(visited) * ratio))
        selected_neg = random.sample(unvisited, min(num_neg, len(unvisited)))
        for menu_id in selected_neg:
            neg_rows.append({'user_id': user_id, 'menu_id': menu_id, 'rating_menu': 0.0})
    return pd.DataFrame(neg_rows)

def process_features(ratings_df, menu_df, rest_df, user_df, cb, cf, gnn, mode):
    # 1. Negative Sampling
    neg_data = generate_negative_samples(ratings_df, menu_df, ratio=1)
    data = pd.concat([ratings_df, neg_data], ignore_index=True)

    # 2. Merge Meta Data
    data = pd.merge(data, menu_df[['menu_id', 'rest_id', 'price']], on='menu_id', how='left')
    rest_temp = rest_df[['rest_id', 'Latitude', 'Longitude', 'rating']].rename(columns={'rating': 'rating_rest'})
    data = pd.merge(data, rest_temp, on='rest_id', how='left')
    
    # Fill NA
    data['price'] = data['price'].fillna(data['price'].mean())
    data['rating_rest'] = data['rating_rest'].fillna(3.0)
    data['Latitude'] = data['Latitude'].fillna(37.2963)
    data['Longitude'] = data['Longitude'].fillna(126.9706)

    # 3. Calculate Scores
    user_pref_dict = user_df.set_index('user_id')['preference'].to_dict()
    
    # CB Score
    data['CB_Score'] = data.apply(lambda row: cb.get_single_cb_score(row['menu_id'], user_pref_dict.get(row['user_id'], "")), axis=1)
    
    # CF Score
    data['CF_Score'] = data.apply(lambda row: cf.model.predict(uid=row['user_id'], iid=row['menu_id']).est, axis=1)

    # Graph Score
    if mode == 'proposed' and gnn:
        data['Graph_Score'] = data.apply(lambda row: gnn.get_graph_score(row['user_id'], row['menu_id']), axis=1).fillna(0)
        g_min, g_max = data['Graph_Score'].min(), data['Graph_Score'].max()
        if g_max > g_min: data['Graph_Score'] = (data['Graph_Score'] - g_min) / (g_max - g_min)
    
    # Normalization
    data['CB_Score'] = data['CB_Score'].clip(0, 1)
    data['CF_Score'] = data['CF_Score'] / 5.0
    data['price'] = np.log1p(data['price'])
    data['Distance_Score'] = 0.5 # temporarily fix the value 
    data['rating_rest'] = data['rating_rest'] / 5.0
    
    # 4. Create X, Y
    if mode == 'baseline':
        X = data[['CB_Score', 'CF_Score', 'price', 'Distance_Score', 'rating_rest']].values
    elif mode == 'proposed':
        X = data[['CB_Score', 'CF_Score', 'Graph_Score', 'price', 'Distance_Score', 'rating_rest']].values
    else: # gnn_only
        X = None 
        
    Y = data['rating_menu'].values
    return X, Y

def run_cv(args, menu_df, rest_df, ratings_df, user_df):
    print(f"🚀 [Cross Validation] Mode: {args.mode} | K={args.k_fold}")
    kf = KFold(n_splits=args.k_fold, shuffle=True, random_state=42)
    scores = []

    # [수정] cf.train() 제거 (생성자에서 이미 학습됨)
    cb = ContentBasedRecommender(DATA_PATHS['menu'])
    cf = CollaborativeRecommender(DATA_PATHS['rating'], DATA_PATHS['menu'])
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(ratings_df)):
        print(f"   Fold {fold+1}/{args.k_fold} processing...")
        train_ratings = ratings_df.iloc[train_idx].copy()

        # 1. Train GNN
        gnn = None
        if args.mode in ['proposed', 'gnn_only']:
            train_ratings.to_csv('temp_train_ratings.csv', index=False)
            gnn = GraphRecommender('temp_train_ratings.csv', DATA_PATHS['menu'])
            gnn.train(embedding_dim=args.gnn_dim, epochs=args.gnn_epochs, lr=args.gnn_lr)
        
        # 2. Train MLP
        if args.mode in ['baseline', 'proposed']:
            X_train, Y_train = process_features(train_ratings, menu_df, rest_df, user_df, cb, cf, gnn, args.mode)
            mlp = MLPBlender(input_dim=X_train.shape[1])
            history = mlp.train(X_train, Y_train, epochs=args.mlp_epochs, batch_size=16, log_interval=9999)
            
            best_val_rmse = min(history.history['val_root_mean_squared_error'])
            scores.append(best_val_rmse)
            print(f"      Best Val RMSE: {best_val_rmse:.4f}")
        else:
            scores.append(0.0) 
        
    avg_score = np.mean(scores)
    print(f"🏁 CV Result ({args.mode}) -> Average RMSE: {avg_score:.4f}")
    return avg_score

def train_final(args, menu_df, rest_df, ratings_df, user_df):
    print(f"🔥 [Final Training] Mode: {args.mode} | Saving to {args.model_name}")
    
    # [수정] cf.train() 제거
    cb = ContentBasedRecommender(DATA_PATHS['menu'])
    cf = CollaborativeRecommender(DATA_PATHS['rating'], DATA_PATHS['menu'])
    
    gnn = None
    if args.mode in ['proposed', 'gnn_only']:
        gnn = GraphRecommender(DATA_PATHS['rating'], DATA_PATHS['menu'])
        gnn.train(embedding_dim=args.gnn_dim, epochs=args.gnn_epochs, lr=args.gnn_lr, verbose=True)
        os.makedirs(MODEL_DIR, exist_ok=True)
        gnn.save_model(os.path.join(MODEL_DIR, f"{args.model_name}_gnn.pth"))

    if args.mode in ['baseline', 'proposed']:
        X, Y = process_features(ratings_df, menu_df, rest_df, user_df, cb, cf, gnn, args.mode)
        mlp = MLPBlender(input_dim=X.shape[1])
        mlp.train(X, Y, epochs=args.mlp_epochs, batch_size=16, log_interval=50)
        
        os.makedirs(MODEL_DIR, exist_ok=True)
        mlp.model.save(os.path.join(MODEL_DIR, f"{args.model_name}_mlp.keras"))
        mlp.save_scaler(os.path.join(MODEL_DIR, f"{args.model_name}_scaler.joblib"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['baseline', 'proposed', 'gnn_only'], required=True)
    parser.add_argument('--job', type=str, choices=['cv', 'train'], required=True)
    parser.add_argument('--k_fold', type=int, default=3)
    parser.add_argument('--model_name', type=str, default='model')
    
    # Hyperparameters
    parser.add_argument('--gnn_epochs', type=int, default=2000)
    parser.add_argument('--gnn_lr', type=float, default=0.005)
    parser.add_argument('--gnn_dim', type=int, default=16)
    parser.add_argument('--mlp_epochs', type=int, default=300)
    
    args = parser.parse_args()
    set_seeds()

    ratings_df = pd.read_csv(DATA_PATHS['rating'])
    if 'rating' in ratings_df.columns: ratings_df.rename(columns={'rating': 'rating_menu'}, inplace=True)
    menu_df = pd.read_csv(DATA_PATHS['menu'])
    rest_df = pd.read_csv(DATA_PATHS['rest'])
    user_df = pd.read_csv(DATA_PATHS['user'])

    if args.job == 'cv':
        run_cv(args, menu_df, rest_df, ratings_df, user_df)
    else:
        train_final(args, menu_df, rest_df, ratings_df, user_df)

    if os.path.exists('temp_train_ratings.csv'): os.remove('temp_train_ratings.csv')

if __name__ == "__main__":
    main()