import faiss

def create_faiss_index(embeddings, dim = None):
    if dim is None:
        dim = embeddings.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.add(embeddings)
    return index

def add_embeddings_to_index(index, embeddings):
    index.add(embeddings)
    return index

def search_faiss_index(index, query_embedding, top_k=5):
    distances, indices = index.search(query_embedding, top_k)
    return distances, indices
