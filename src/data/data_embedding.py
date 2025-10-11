import faiss
from src.config.config import EMBEDDER_MODEL_NAME
from sentence_transformers import SentenceTransformer
from src.database.faiss_interface import add_embeddings_to_index
import numpy as np

def get_model(model_name : str = EMBEDDER_MODEL_NAME) -> SentenceTransformer:
    model = SentenceTransformer(model_name)
    return model

def embed_texts(texts: list[str], model: SentenceTransformer, 
                index: faiss.Index = None) -> np.ndarray:
    embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()
    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    if index is not None:
        add_embeddings_to_index(index, embeddings)
    return embeddings
