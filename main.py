import pandas as pd
from src.search.candidate_generator import CandidateGenerator
from src.search.contextual_bandit import ContextualBanditReranker
from src.utils.items_data import ItemsData

data = pd.read_csv('data/data_raw.csv')

items_data = ItemsData(df=data)
cg = CandidateGenerator(alpha=0.05, items_data=items_data)

query = "teamsync"

candidates = cg.generate_candidates(query, top_k=50)

cb = ContextualBanditReranker(candidates=candidates, items_data=items_data)

cb.rerank(query)

candidates = cb.get_page(0, 3)

print("Top candidates:")
for idx, score in candidates:
    print(f"Index: {idx}, Score: {score}")
    print("-----")
    print(data.iloc[idx]['Name'])
    print("-----")
    print(data.iloc[idx]['Description'])
    print("===============================================================")
