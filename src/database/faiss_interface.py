import faiss
import numpy as np

def create_faiss_index(embeddings: np.ndarray, dim: int = None) -> faiss.Index:
    if dim is None:
        dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.add(embeddings)
    return index

def add_embeddings_to_index(index: faiss.Index, embeddings: np.ndarray) -> faiss.Index:
    index.add(embeddings)
    return index

def search_faiss_index(index: faiss.Index, query_embedding: np.ndarray, top_k: int = 5) -> tuple:
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices
