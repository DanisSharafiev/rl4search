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
    """Compute Normalized Discounted Cumulative Gain (NDCG) at rank k.

    Args:
        relevance_scores (List[float]): A list of relevance scores in the order of predicted ranking.
        k (int): The rank position up to which NDCG is computed.

    Returns:
        float: The NDCG value at rank k.
    """
    def dcg(scores: List[int], k: int) -> float:
        return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(scores[:k]))

    ideal_relevance = sorted(relevance_scores, reverse=True)
    actual_dcg = dcg(relevance_scores, k)
    ideal_dcg = dcg(ideal_relevance, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

def calculate_delta_ndcg_rewards(
    displayed_items: List[tuple[int, float]], 
    user_clicks_df: pd.DataFrame,
    user_id: int,
    baseline_candidates: List[tuple[int, float]],
    scale_factor: float = 10.0
) -> List[tuple[int, float]]:
    """
    Calculates rewards for a batch of items based on the difference between 
    the session's NDCG and a baseline NDCG.

    Args:
        displayed_items: List of (item_idx, score) tuples shown to the user by the bandit.
        user_clicks_df: DataFrame containing all user clicks.
        user_id: The ID of the current user.
        baseline_candidates: List of (item_idx, score) tuples from the baseline model (e.g. CandidateGenerator).
        scale_factor: Multiplier for the reward to make the signal stronger for the bandit.

    Returns:
        List of (item_idx, reward) tuples ready for ContextualBanditReranker.update_batch.
    """
    # 1. Calculate Baseline NDCG
    # We assume the baseline shows the same number of items as the bandit
    k = len(displayed_items)
    baseline_top_k = baseline_candidates[:k]
    
    baseline_clicks = get_serp_clicks(user_id, user_clicks_df, baseline_top_k)
    baseline_ndcg = ndcg(baseline_clicks, k=k)

    # 2. Calculate Bandit NDCG
    bandit_clicks = get_serp_clicks(user_id, user_clicks_df, displayed_items)
    bandit_ndcg = ndcg(bandit_clicks, k=k)

    # 3. Calculate Delta Reward
    delta = bandit_ndcg - baseline_ndcg
    reward = delta * scale_factor
    
    # Assign the same session-level reward to all items in the session
    updates = [(item_idx, reward) for item_idx, _ in displayed_items]
    return updates
