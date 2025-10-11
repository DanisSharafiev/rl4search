import pandas as pd

raw_df = pd.read_csv('data/data_raw.csv')

raw_df.head()

import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from config.config import TEXT_COLUMN

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')

from src.data.data_preprocess import preprocess_text_for_BM25, build_bm25_index_from_df, preprocess_text

df = preprocess_text_for_BM25(raw_df, text_col=TEXT_COLUMN)
bm25, docs, df = build_bm25_index_from_df(df)

# docs[:1]

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

query = "AI game russian"

query = preprocess_text(query, stop_words, lemmatizer)

bm25.get_top_n(query, docs, n=5)

from src.database.faiss_interface import create_faiss_index
from src.data.data_embedding import get_model, embed_texts

model = get_model()

embeddings = embed_texts(df['Description'].tolist(), model)

index = create_faiss_index(embeddings, dim=embeddings.shape[1])
print(embeddings[:5])
