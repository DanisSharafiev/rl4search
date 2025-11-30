import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.utils.items_data import ItemsData

class RewardPredictor(nn.Module):
    def __init__(self, input_dim):
        super(RewardPredictor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        return self.net(x)

class NeuralBanditReranker:
    def __init__(self, candidates, items_data: ItemsData, model=None, optimizer=None, epsilon=0.1, lr=0.001):
        self.candidates = candidates
        self.items_data = items_data
        self.epsilon = epsilon
        self.embedding_size = items_data.embedding_size
        self.input_dim = self.embedding_size * 2 + 1
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        
        if model:
            self.model = model
        else:
            self.model = RewardPredictor(self.input_dim).to(self.device)
            
        if optimizer:
            self.optimizer = optimizer
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
            
        self.criterion = nn.MSELoss()
        self.ranked_items = []

    def _get_features(self, query_emb, item_idx):
        item_emb = self.items_data.index.reconstruct(int(item_idx))
        
        # Explicit interaction feature: Cosine Similarity
        # Helps the model generalize better than just raw concatenation
        norm_q = np.linalg.norm(query_emb)
        norm_i = np.linalg.norm(item_emb)
        if norm_q > 0 and norm_i > 0:
            cos_sim = np.dot(query_emb, item_emb) / (norm_q * norm_i)
        else:
            cos_sim = 0.0
            
        features = np.concatenate([query_emb, item_emb, [cos_sim]])
        return torch.FloatTensor(features).to(self.device)

    def rerank(self, query: str) -> list:
        query_emb = self.items_data.model.encode([query], convert_to_tensor=True).cpu().numpy().flatten()
        
        self.model.eval()
        scores = []
        
        with torch.no_grad():
            for item_idx, _ in self.candidates:
                features = self._get_features(query_emb, item_idx)
                pred_reward = self.model(features).item()
                scores.append((item_idx, pred_reward))
        
        if np.random.random() < self.epsilon:
            np.random.shuffle(scores)
        else:
            scores.sort(key=lambda x: x[1], reverse=True)
            
        self.ranked_items = scores
        return self.ranked_items

    def get_page(self, idx: int, k: int) -> list:
        if not self.ranked_items:
            return []
        start = idx * k
        end = min((idx + 1) * k, len(self.ranked_items))
        return self.ranked_items[start:end]

    def update_batch(self, query: str, updates: list[tuple[int, float]]) -> None:
        query_emb = self.items_data.model.encode([query], convert_to_tensor=True).cpu().numpy().flatten()
        
        self.model.train()
        self.optimizer.zero_grad()
        
        batch_features = []
        batch_rewards = []
        
        for item_idx, reward in updates:
            features = self._get_features(query_emb, item_idx)
            batch_features.append(features)
            batch_rewards.append(reward)
            
        if not batch_features:
            return

        inputs = torch.stack(batch_features)
        targets = torch.FloatTensor(batch_rewards).to(self.device).unsqueeze(1)
        
        predictions = self.model(inputs)
        loss = self.criterion(predictions, targets)
        
        loss.backward()
        self.optimizer.step()

    def update_pairwise(self, query: str, displayed_items: list[tuple[int, float]], clicks: list[int]) -> None:
        query_emb = self.items_data.model.encode([query], convert_to_tensor=True).cpu().numpy().flatten()
        
        pos_items = [item_idx for (item_idx, _), clicked in zip(displayed_items, clicks) if clicked == 1]
        neg_items = [item_idx for (item_idx, _), clicked in zip(displayed_items, clicks) if clicked == 0]
        
        if not pos_items or not neg_items:
            return

        self.model.train()
        self.optimizer.zero_grad()
        
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        pairs_count = 0
        
        for pos_idx in pos_items:
            pos_features = self._get_features(query_emb, pos_idx)
            pos_score = self.model(pos_features)
            
            for neg_idx in neg_items:
                neg_features = self._get_features(query_emb, neg_idx)
                neg_score = self.model(neg_features)
                
                current_loss = torch.relu(1.0 - (pos_score - neg_score))
                loss = loss + current_loss
                pairs_count += 1
        
        if pairs_count > 0:
            loss = loss / pairs_count
            loss.backward()
            self.optimizer.step()
