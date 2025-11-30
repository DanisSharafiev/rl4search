from src.utils.items_data import ItemsData
import numpy as np

# TODO: Implement regularization for A and b to avoid singularity and overfitting

class ContextualBanditReranker:
    def __init__(self, candidates, items_data: ItemsData, A=None, b=None, regularization = 0.1, small_sigma=0.1) -> None:
        self.candidates = candidates
        self.items_data = items_data
        self.regularization = regularization
        self.small_sigma = small_sigma
        
        if A is not None and b is not None:
            self.A = A
            self.b = b
        else:
            self.A = self.regularization * np.eye(items_data.embedding_size)
            self.b = np.zeros((items_data.embedding_size, 1))
            
        self.ranked_items = []

    def _get_context(self, query_emb, item_idx):
        item_embedding = self.items_data.index.reconstruct(int(item_idx))
        return query_emb * item_embedding

    def update_model(self, query: str, item_idx : int, reward : float) -> None:
        query_emb = self.items_data.model.encode([query], convert_to_tensor=True).cpu().numpy().flatten()
        x = self._get_context(query_emb, item_idx).reshape(-1, 1)
        self.A += np.outer(x, x)
        self.b += reward * x
    
    def update_batch(self, query: str, updates: list[tuple[int, float]]) -> None:
        query_emb = self.items_data.model.encode([query], convert_to_tensor=True).cpu().numpy().flatten()
        
        for item_idx, reward in updates:
            x = self._get_context(query_emb, item_idx).reshape(-1, 1)
            self.A += np.outer(x, x)
            self.b += reward * x
    
    def rerank(self, query : str) -> list:
        query_emb = self.items_data.model.encode([query], convert_to_tensor=True).cpu().numpy().flatten()
        
        A_inv = np.linalg.inv(self.A)
        theta_mean = A_inv.dot(self.b).flatten()
        
        sigma = (self.small_sigma ** 2) * A_inv
        theta_sample = np.random.multivariate_normal(theta_mean, sigma)

        scores = []
        for item_idx, _ in self.candidates:
            context = self._get_context(query_emb, item_idx)
            score = np.dot(theta_sample, context)
            scores.append((item_idx, float(score)))

        self.ranked_items = sorted(scores, key=lambda x: x[1], reverse=True)
        return self.ranked_items
    
    def get_page(self, idx : int, k : int) -> list:
        if not self.ranked_items:
            print("No ranked items available. Please run rerank() first.")
            return []

        if idx * k >= len(self.ranked_items):
            print("Index out of range")
            return []

        k_items = self.ranked_items[idx * k:min((idx + 1) * k, len(self.ranked_items))]
        return k_items
