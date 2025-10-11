from src.database.faiss_interface import create_faiss_index, search_faiss_index
from src.config.config import TEXT_COLUMN
from src.data.data_embedding import get_model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.utils.similarity_metrics import cosine_similarity
from src.data.data_preprocess import preprocess_text, build_bm25_index_from_df, preprocess_text_for_BM25

class CandidateGenerator:
    def __init__(self, alpha=0.5, df=None):
        if df is None:
            raise ValueError("DataFrame must be provided")
        self.alpha = alpha
        df = preprocess_text_for_BM25(df, text_col=TEXT_COLUMN)
        print("Preprocessing done.")
        self.bm25, self.docs, self.df = build_bm25_index_from_df(df)
        print("BM25 index built.")
        self.model = get_model()
        print("Embedding model loaded.")
        embeddings = self.model.encode(self.df[TEXT_COLUMN].tolist(), convert_to_tensor=True).cpu().numpy()
        print("Embeddings generated.")
        self.index = create_faiss_index(embeddings)
        print("FAISS index created.")
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        print("CandidateGenerator initialized.")

    def generate_candidates(self, query, top_k=5):
        query_processed = preprocess_text(query, self.stop_words, self.lemmatizer)

        # BM25 part
        bm25_scores = self.bm25.get_scores(query_processed)

        # normalization
        bm25_min = bm25_scores.min()
        bm25_max = bm25_scores.max()
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0
        bm25_scores = (bm25_scores - bm25_min) / bm25_range

        bm25_top_indices = bm25_scores.argsort()[::-1][:top_k]
        bm25_candidates = [(idx, bm25_scores[idx]) for idx in bm25_top_indices]

        # Embedding part
        query_embedding = self.model.encode([query], convert_to_tensor=True).cpu().numpy()
        distances, indices = search_faiss_index(self.index, query_embedding, top_k=top_k)
        embedding_candidates = [(idx, 1 - distances[0][i]) for i, idx in enumerate(indices[0])]

        result = {}

        all_indices = set([idx for idx, _ in bm25_candidates] + [idx for idx, _ in embedding_candidates])

        # Combine and rank candidates
        for idx in all_indices:
            bm25_score = bm25_scores[idx]
            
            doc_embedding = self.index.reconstruct(int(idx))
            embedding_score = cosine_similarity(query_embedding[0], doc_embedding)
            
            result[idx] = self.alpha * bm25_score + (1 - self.alpha) * embedding_score

        result = sorted(result.items(), key=lambda x: x[1], reverse=True)

        return result[:top_k]