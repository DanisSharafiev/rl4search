from typing import List
import math
import pandas as pd

def get_serp_clicks(
             user_id : int,                                      # ID of the user
             user_clicks : pd.DataFrame,                         # dataframe with user clicks
             serp : List[int] | List[tuple[int, float]]          # list of project IDs in the SERP
             ) -> List[int]:
    result = []
    if serp and not isinstance(serp[0], int):
        serp = [proj_id for proj_id, _ in serp]
    for project_id in serp:
        if not user_clicks[(user_clicks['user_id'] == user_id) & (user_clicks['project_id'] == project_id)].empty:
            result.append(1)
        else:
            result.append(0)
    return result

def ndcg(relevance_scores: List[int], k : int) -> float:
    def dcg(scores: List[int], k: int) -> float:
        return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(scores[:k]))

    ideal_relevance = sorted(relevance_scores, reverse=True)
    actual_dcg = dcg(relevance_scores, k)
    ideal_dcg = dcg(ideal_relevance, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def precision_at_k(relevance_scores: List[int], k: int) -> float:
    if k == 0:
        return 0.0
    relevant_items = sum(relevance_scores[:k])
    return relevant_items / k

def calculate_delta_ndcg_rewards(
    displayed_items: List[tuple[int, float]], 
    user_clicks_df: pd.DataFrame,
    user_id: int,
    baseline_candidates: List[tuple[int, float]],
    scale_factor: float = 10.0
) -> List[tuple[int, float]]:
    k = len(displayed_items)
    baseline_top_k = baseline_candidates[:k]
    
    baseline_clicks = get_serp_clicks(user_id, user_clicks_df, baseline_top_k)
    baseline_ndcg = ndcg(baseline_clicks, k=k)

    bandit_clicks = get_serp_clicks(user_id, user_clicks_df, displayed_items)
    bandit_ndcg = ndcg(bandit_clicks, k=k)

    delta = bandit_ndcg - baseline_ndcg
    reward = delta * scale_factor
    
    updates = [(item_idx, reward) for item_idx, _ in displayed_items]
    return updates
