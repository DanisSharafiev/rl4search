from src.database.faiss_interface import create_faiss_index, search_faiss_index
from src.config.config import TEXT_COLUMN
from src.data.data_embedding import get_model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from src.utils.similarity_metrics import cosine_similarity
from src.data.data_preprocess import build_bm25_index_from_df, preprocess_text_for_BM25
import pandas as pd

class ItemsData:
    def __init__(self, df : pd.DataFrame = None) -> None:
        if df is None:
            raise ValueError("DataFrame must be provided")
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
        self.embedding_size = embeddings.shape[1]