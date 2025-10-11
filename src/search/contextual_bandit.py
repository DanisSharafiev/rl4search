from src.utils.items_data import ItemsData
import numpy as np

# TODO: Implement regularization for A and b to avoid singularity and overfitting

class ContextualBanditReranker:
    def __init__(self, candidates, items_data: ItemsData) -> None:
        self.candidates = candidates
        self.items_data = items_data
        self.regularization = 0.1
        self.small_sigma = 0.1
        self.A = self.regularization * np.eye(items_data.embedding_size)
        self.b = np.zeros((items_data.embedding_size, 1))
        self.ranked_items = []

    def update_model(self, item_idx : int, reward : float) -> None:
        x = self.items_data.index.reconstruct(int(item_idx)).reshape(-1, 1)
        self.A += np.outer(x, x)
        self.b += reward * x

    def reward_function(self, query : str, item_idx : int) -> float:
        theta_mean = np.linalg.inv(self.A).dot(self.b).flatten()
        sigma = (self.small_sigma ** 2) * np.linalg.inv(self.A)
        theta_sample = np.random.multivariate_normal(theta_mean, sigma)
        query_embedding = self.items_data.model.encode([query], convert_to_tensor=True).cpu().numpy().flatten()

        item_embedding = self.items_data.index.reconstruct(int(item_idx))

        context = query_embedding * item_embedding

        reward = np.dot(theta_sample, context)

        return float(reward)
    
    def rerank(self, query : str) -> list:
        print(self.candidates[:2])
        self.ranked_items = sorted(self.candidates, key=lambda x: self.reward_function(query, x[0]), reverse=True)
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
