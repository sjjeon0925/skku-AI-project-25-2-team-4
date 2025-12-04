import argparse
import pandas as pd
import numpy as np
import os
import tensorflow as tf
from filtering.contents_based import ContentBasedRecommender
from filtering.collaborative import CollaborativeRecommender
from filtering.blender_mlp import MLPBlender
from filtering.graph_model import GraphRecommender
from utils import DATA_PATHS, calculate_distance_score, MODEL_DIR

def ndcg_at_k(r, k):
    def dcg(r, k):
        r = np.asfarray(r)[:k]
        if r.size: return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        return 0.
    dcg_max = dcg(sorted(r, reverse=True), k)
    if not dcg_max: return 0.
    return dcg(r, k) / dcg_max

def evaluate(args):
    # Load Models
    cb = ContentBasedRecommender(DATA_PATHS['menu'])
    cf = CollaborativeRecommender(DATA_PATHS['rating'], DATA_PATHS['menu'])
    
    gnn = None
    if args.mode in ['proposed', 'gnn_only']:
        gnn = GraphRecommender(DATA_PATHS['rating'], DATA_PATHS['menu'])
        gnn.load_model(os.path.join(MODEL_DIR, f"{args.model_name}_gnn.pth"))

    mlp = None
    if args.mode in ['baseline', 'proposed']:
        input_dim = 5 if args.mode == 'baseline' else 6
        mlp = MLPBlender(input_dim)
        mlp.model = tf.keras.models.load_model(os.path.join(MODEL_DIR, f"{args.model_name}_mlp.keras"))
        mlp.load_scaler(os.path.join(MODEL_DIR, f"{args.model_name}_scaler.joblib"))

    # Load Data & Targets
    ratings_df = pd.read_csv(DATA_PATHS['rating'])
    good_ratings = ratings_df[ratings_df['rating'] >= 4.0]
    user_ground_truth = good_ratings.groupby('user_id')['menu_id'].apply(list).to_dict()
    
    menu_df = pd.read_csv(DATA_PATHS['menu'])
    rest_df = pd.read_csv(DATA_PATHS['rest'])
    user_df = pd.read_csv(DATA_PATHS['user'])

    # Eval Loop
    metrics = {'recall': [], 'precision': [], 'ndcg': []}
    
    for user_id, true_ids in user_ground_truth.items():
        user_row = user_df[user_df['user_id'].astype(str) == str(user_id)]
        if user_row.empty: continue
        
        # Candidates
        df = menu_df.copy()
        
        # Features
        df['CB'] = df['menu_id'].apply(lambda x: cb.get_single_cb_score(x, user_row['preference'].iloc[0]))
        df['CF'] = df['menu_id'].apply(lambda x: cf.model.predict(uid=user_id, iid=x).est)
        
        if gnn:
            df['GNN'] = df['menu_id'].apply(lambda x: gnn.get_graph_score(user_id, x))
            # Norm
            gmin, gmax = df['GNN'].min(), df['GNN'].max()
            if gmax > gmin: df['GNN'] = (df['GNN'] - gmin) / (gmax - gmin)
        
        # Meta
        df = pd.merge(df, rest_df[['rest_id', 'Latitude', 'Longitude', 'rating']], on='rest_id', how='left')
        df['Dist'] = df.apply(lambda r: calculate_distance_score('b', r['Latitude'], r['Longitude']), axis=1)
        df['rating'] = df['rating'].fillna(3.0)

        # Predict
        if args.mode == 'gnn_only':
            df['Score'] = df['GNN']
        elif args.mode == 'baseline':
            X = df[['CB', 'CF', 'price', 'Dist', 'rating']].values
            df['Score'] = mlp.predict(X)
        else: # proposed
            X = df[['CB', 'CF', 'GNN', 'price', 'Dist', 'rating']].values
            df['Score'] = mlp.predict(X)

        # Top-K
        recs = df.sort_values(by='Score', ascending=False).head(10)['menu_id'].tolist()
        
        # Calc Metrics
        hits = set(true_ids) & set(recs)
        metrics['recall'].append(len(hits) / len(true_ids))
        metrics['precision'].append(len(hits) / 10)
        metrics['ndcg'].append(ndcg_at_k([1 if m in true_ids else 0 for m in recs], 10))

    print(f"\n📊 Result [{args.mode}]:")
    print(f"   Recall@10: {np.mean(metrics['recall'])*100:.2f}%")
    print(f"   Prec@10  : {np.mean(metrics['precision'])*100:.2f}%")
    print(f"   NDCG@10  : {np.mean(metrics['ndcg']):.4f}")
    
    # 결과 리턴용 (자동화 스크립트가 잡을 수 있게)
    return np.mean(metrics['recall'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True)
    parser.add_argument('--model_name', type=str, default='model')
    args = parser.parse_args()
    evaluate(args)