import pandas as pd
from src.search.candidate_generator import CandidateGenerator
from src.search.contextual_bandit import ContextualBanditReranker
from src.search.cross_encoder_reranker import CrossEncoderReranker
from src.utils.items_data import ItemsData

data = pd.read_csv('data/data_raw.csv')

items_data = ItemsData(df=data)
cg = CandidateGenerator(alpha=0.05, items_data=items_data)

query = "teamsync"

candidates = cg.generate_candidates(query, top_k=50)

# Example: use contextual bandit reranker
cb = ContextualBanditReranker(candidates=candidates, items_data=items_data)
cb.rerank(query)
cb_candidates = cb.get_page(0, 3)

# Example: use Cross-Encoder reranker (prototype). This will load a cross-encoder
# model and rerank the candidate set (works best when candidates size is modest).
cer = CrossEncoderReranker(candidates=candidates, items_data=items_data, batch_size=16)
cer.rerank(query, top_k=50)
ce_candidates = cer.get_page(0, 3)

# Choose which results to print (switch between `cb_candidates` and `ce_candidates`)
candidates = ce_candidates

print("Top candidates:")
for idx, score in candidates:
    print(f"Index: {idx}, Score: {score}")
    print("-----")
    print(data.iloc[idx]['Name'])
    print("-----")
    print(data.iloc[idx]['Description'])
    print("===============================================================")
