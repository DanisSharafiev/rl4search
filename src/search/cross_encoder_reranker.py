from typing import Optional
import torch
from sentence_transformers import CrossEncoder
from src.config.config import TEXT_COLUMN
from src.utils.items_data import ItemsData

def _choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

class CrossEncoderReranker:
    def __init__(self, 
                 candidates, 
                 items_data: 
                 ItemsData, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: Optional[str] = None, 
                 batch_size: int = 32
                 ) -> None:
        if items_data is None:
            raise ValueError("ItemsData instance must be provided")
        self.candidates = candidates
        self.items_data = items_data
        self.model_name = model_name
        self.device = device or _choose_device()
        self.batch_size = batch_size
        self.model = CrossEncoder(self.model_name, device=self.device)
        self.ranked_items = []

    def rerank(self, 
               query: str, 
               top_k: int | None = None
               ) -> list:
        if top_k is None:
            candidate_list = self.candidates
        else:
            candidate_list = self.candidates[:top_k]

        pairs = []
        idxs = []
        for idx, _ in candidate_list:
            text = self.items_data.df.iloc[int(idx)][TEXT_COLUMN]
            pairs.append([query, str(text)])
            idxs.append(int(idx))

        if not pairs:
            self.ranked_items = []
            return self.ranked_items

        scores = self.model.predict(pairs, batch_size=self.batch_size)

        ranked = sorted(zip(idxs, scores), key=lambda x: x[1], reverse=True)
        self.ranked_items = ranked
        return self.ranked_items

    def get_page(self, idx: int, k: int) -> list:
        if not self.ranked_items:
            print("No ranked items available. Please run rerank() first.")
            return []

        if idx * k >= len(self.ranked_items):
            print("Index out of range")
            return []

        return self.ranked_items[idx * k:min((idx + 1) * k, len(self.ranked_items))]
