from src.utils.items_data import ItemsData

# regularization for discounting

class ContextualBanditReranker:
    def __init__(self, candidates, items_data: ItemsData):
        self.candidates = candidates
        self.items_data = items_data
        self.theta = [0.0] * len(candidates)

    def reward_function(self, query):
        # TODO: Implement a proper reward function based on relevance
        # Linear function
        query_embedding = self.items_data.model.encode([query], convert_to_tensor=True).cpu().numpy()

        return 1.0
    
    def rerank(self, query : str) -> list:
        self.ranked_items = sorted(self.candidates, key=lambda x: self.reward_function(query, x), reverse=True)
        return self.ranked_items
    
    def get_k_items(self, idx : int, k : int) -> list:
        if not self.ranked_items:
            print("No ranked items available. Please run rerank() first.")
            return []

        if idx * k >= len(self.ranked_items):
            print("Index out of range")
            return []

        k_items = self.ranked_items[idx * k:max((idx + 1) * k, len(self.ranked_items))]
        return k_items