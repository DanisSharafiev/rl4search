from src.config.config import EMBEDDER_MODEL_NAME
from sentence_transformers import SentenceTransformer
from src.database.faiss_interface import add_embeddings_to_index

def get_model(model_name=EMBEDDER_MODEL_NAME) -> tuple:
    model = SentenceTransformer(model_name)
    return model

def embed_texts(texts, model, index = None) -> tuple:
    embeddings = model.encode(texts, convert_to_tensor=True).cpu().numpy()
    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
    if index is not None:
        add_embeddings_to_index(embeddings, index)
    return embeddings
