import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi
from src.config.config import TEXT_COLUMN
import pandas as pd

nltk.download('punkt')
nltk.download('wordnet')

def preprocess_text(text : str, stop_words: set, lemmatizer: WordNetLemmatizer) -> list[str]:
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word for word in tokens if word.lower() not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    tokens = [word.lower() for word in tokens]
    return tokens

def preprocess_text_for_BM25(raw_df: pd.DataFrame, text_col: str = 'text', 
                             processed_col: str = 'processed_text') -> pd.DataFrame:
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    df = raw_df.copy()
    df[processed_col] = df[text_col].apply(lambda x: ' '.join(preprocess_text(x, stop_words, lemmatizer)))
    return df

def build_bm25_index_from_df(df: pd.DataFrame, processed_col: str = TEXT_COLUMN) -> tuple:
    docs = []
    for text in df[processed_col]:
        docs.append(text.split())
    bm25 = BM25Okapi(docs)
    return bm25, docs, df
