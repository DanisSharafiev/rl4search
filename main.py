import pandas as pd
from src.candidate_generation.candidate_generator import CandidateGenerator

data = pd.read_csv('data/data_raw.csv')

cg = CandidateGenerator(alpha=0.05, df=data)

query = "teamsync"

candidates = cg.generate_candidates(query, top_k=5)

print("Top candidates:")
for idx, score in candidates:
    print(f"Index: {idx}, Score: {score}")
    print("-----")
    print(data.iloc[idx]['Name'])
    print("-----")
    print(data.iloc[idx]['Description'])
    print("===============================================================")
