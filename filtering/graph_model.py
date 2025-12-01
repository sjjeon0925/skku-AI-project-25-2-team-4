import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
import os

EMBEDDING_STD = 0.01
GNN_EPOCHS = 500
GNN_LR = 0.005
GNN_WD = 1e-6

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=16, n_layers=3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers

        # 초기 임베딩 (0: User, 1: Item)
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # 임베딩 초기화 (Normal Distribution)
        nn.init.normal_(self.user_embedding.weight, std=EMBEDDING_STD)
        nn.init.normal_(self.item_embedding.weight, std=EMBEDDING_STD)

    def forward(self, adj_matrix):
        # 1. 초기 임베딩 결합
        all_emb = torch.cat([self.user_embedding.weight, self.item_embedding.weight])
        embs = [all_emb]

        # 2. 레이어 전파 (Graph Convolution)
        # LightGCN은 활성화 함수 없이 이웃 정보를 평균(또는 가중합) 냄
        for _ in range(self.n_layers):
            all_emb = torch.sparse.mm(adj_matrix, all_emb)
            embs.append(all_emb)
        
        # 3. 모든 레이어의 임베딩을 평균 (Readout)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items

class GraphRecommender:
    def __init__(self, ratings_path='./data/rating_data.csv', menu_path='./data/menu_data.csv'):
        self.ratings_path = ratings_path
        
        # 데이터 로드 및 매핑 생성
        ratings_df = pd.read_csv(ratings_path)
        menu_df = pd.read_csv(menu_path)
        
        self.user_ids = ratings_df['user_id'].unique()
        self.menu_ids = menu_df['menu_id'].unique()
        
        # ID <-> Index 매핑
        self.user2idx = {uid: i for i, uid in enumerate(self.user_ids)}
        self.idx2user = {i: uid for i, uid in enumerate(self.user_ids)}
        self.menu2idx = {mid: i for i, mid in enumerate(self.menu_ids)}
        self.idx2menu = {i: mid for i, mid in enumerate(self.menu_ids)}
        
        self.num_users = len(self.user_ids)
        self.num_items = len(self.menu_ids)
        
        self.model = None
        self.final_user_emb = None
        self.final_item_emb = None
        self.adj_matrix = None

    def _build_adj_matrix(self, df):
        # 상호작용이 있는 노드 쌍 추출
        src = [self.user2idx[u] for u in df['user_id']]
        dst = [self.menu2idx[m] + self.num_users for m in df['menu_id']] # Item 인덱스는 User 뒤에 이어짐
        
        # 양방향 엣지 생성 (User <-> Item)
        row = src + dst
        col = dst + src
        values = torch.ones(len(row))
        
        # 희소 행렬(Sparse Matrix) 생성
        N = self.num_users + self.num_items
        i = torch.LongTensor([row, col])
        adj = torch.sparse_coo_tensor(i, values, (N, N))
        
        # 정규화 (D^-1/2 * A * D^-1/2) - 간단하게 Row Normalize로 대체 가능하나 여기선 생략
        # 실제 성능을 위해선 정규화 과정이 중요합니다.
        return adj.coalesce()

    def train(self):
        epochs = GNN_EPOCHS
        lr = GNN_LR

        print(f"GNN(LightGCN) 모델 학습 시작... (User: {self.num_users}, Item: {self.num_items})")
        
        ratings_df = pd.read_csv(self.ratings_path)
        train_df = ratings_df[ratings_df['rating'] > 0] 
        
        self.adj_matrix = self._build_adj_matrix(train_df)
        self.model = LightGCN(self.num_users, self.num_items)


        # optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=GNN_WD)
        
        # 학습 데이터 준비 (User, Positive Item)
        user_indices = [self.user2idx[u] for u in train_df['user_id']]
        pos_item_indices = [self.menu2idx[m] for m in train_df['menu_id']]
        
        user_indices_t = torch.LongTensor(user_indices)
        pos_item_indices_t = torch.LongTensor(pos_item_indices)
        
        for epoch in range(epochs):
            # 1. Forward
            users_emb, items_emb = self.model(self.adj_matrix)
            
            # 2. Negative Sampling (안 먹은 메뉴 무작위 하나 뽑기)
            # 데이터가 적으므로 간단하게 랜덤으로 뽑음
            neg_item_indices = []
            for u_idx in user_indices:
                # 간단한 구현: 전체 메뉴 중 랜덤 하나 뽑기 (운 좋게 먹은게 걸릴 수 있지만 학습엔 큰 지장 없음)
                # 정석은 안 먹은 것만 골라야 하지만 속도를 위해 랜덤 채택
                rand_item = np.random.randint(0, self.num_items)
                neg_item_indices.append(rand_item)
            
            neg_item_indices_t = torch.LongTensor(neg_item_indices)
            
            # 3. 점수 계산
            # 먹은 거(Positive) 점수
            u_emb = users_emb[user_indices_t]
            pos_i_emb = items_emb[pos_item_indices_t]
            pos_scores = torch.sum(u_emb * pos_i_emb, dim=1)
            
            # 안 먹은 거(Negative) 점수
            neg_i_emb = items_emb[neg_item_indices_t]
            neg_scores = torch.sum(u_emb * neg_i_emb, dim=1)
            
            # 4. BPR Loss 계산
            # (먹은거 점수 - 안먹은거 점수)가 클수록 좋음
            loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch+1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        # 학습 완료 후 최종 임베딩 저장
        with torch.no_grad():
            self.final_user_emb, self.final_item_emb = self.model(self.adj_matrix)
            
        print("GNN 모델 학습 완료.")

    def get_graph_score(self, user_id, menu_id):
        """
        User와 Menu 사이의 그래프 기반 연결 점수(내적) 반환
        """
        if self.model is None:
            return 0.0
            
        # ID가 학습 데이터에 없으면 0 반환
        if user_id not in self.user2idx or menu_id not in self.menu2idx:
            return 0.0
            
        u_idx = self.user2idx[user_id]
        m_idx = self.menu2idx[menu_id]
        
        u_emb = self.final_user_emb[u_idx]
        m_emb = self.final_item_emb[m_idx]
        
        # 내적 (Dot Product)
        score = torch.dot(u_emb, m_emb).item()
        return score

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'user2idx': self.user2idx,
            'menu2idx': self.menu2idx,
            'embeddings': (self.final_user_emb, self.final_item_emb)
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, weights_only=False)
        self.user2idx = checkpoint['user2idx']
        self.menu2idx = checkpoint['menu2idx']
        self.num_users = len(self.user2idx)
        self.num_items = len(self.menu2idx)
        self.idx2user = {v:k for k,v in self.user2idx.items()} # 복원
        self.idx2menu = {v:k for k,v in self.menu2idx.items()} # 복원
        
        self.model = LightGCN(self.num_users, self.num_items)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.final_user_emb, self.final_item_emb = checkpoint['embeddings']