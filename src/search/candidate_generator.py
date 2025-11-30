from src.database.faiss_interface import search_faiss_index
from src.utils.similarity_metrics import cosine_similarity
from src.data.data_preprocess import preprocess_text
from src.utils.items_data import ItemsData

class CandidateGenerator:
    def __init__(self, alpha : float = 0.5, items_data: ItemsData = None) -> None:
        if items_data is None:
            raise ValueError("ItemsData instance must be provided")
        self.items_data = items_data
        self.alpha = alpha
        print("CandidateGenerator initialized.")

    def generate_candidates(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        query_processed = preprocess_text(query, self.items_data.stop_words, self.items_data.lemmatizer)

        # BM25 part
        bm25_scores = self.items_data.bm25.get_scores(query_processed)

        # normalization
        bm25_min = bm25_scores.min()
        bm25_max = bm25_scores.max()
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0
        bm25_scores = (bm25_scores - bm25_min) / bm25_range

        bm25_top_indices = bm25_scores.argsort()[::-1][:top_k]
        bm25_candidates = [(idx, bm25_scores[idx]) for idx in bm25_top_indices]

        # Embedding part
        query_embedding = self.items_data.model.encode([query], convert_to_tensor=True).cpu().numpy()
        distances, indices = search_faiss_index(self.items_data.index, query_embedding, top_k=top_k)
        embedding_candidates = [(idx, 1 - distances[0][i]) for i, idx in enumerate(indices[0])]

        result = {}

        all_indices = set([idx for idx, _ in bm25_candidates] + [idx for idx, _ in embedding_candidates])

        # Combine and rank candidates
        for idx in all_indices:
            bm25_score = bm25_scores[idx]

            doc_embedding = self.items_data.index.reconstruct(int(idx))
            embedding_score = cosine_similarity(query_embedding[0], doc_embedding)
            
            result[idx] = self.alpha * bm25_score + (1 - self.alpha) * embedding_score

        result = sorted(result.items(), key=lambda x: x[1], reverse=True)

        return result[:top_k]