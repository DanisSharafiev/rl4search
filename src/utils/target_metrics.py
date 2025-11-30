from typing import List
import math

def ndcg(relevance_scores: List[float], k: int) -> float:
    """Compute Normalized Discounted Cumulative Gain (NDCG) at rank k.

    Args:
        relevance_scores (List[float]): A list of relevance scores in the order of predicted ranking.
        k (int): The rank position up to which NDCG is computed.

    Returns:
        float: The NDCG value at rank k.
    """
    def dcg(scores: List[float], k: int) -> float:
        return sum((2**rel - 1) / math.log2(idx + 2) for idx, rel in enumerate(scores[:k]))

    ideal_relevance = sorted(relevance_scores, reverse=True)
    actual_dcg = dcg(relevance_scores, k)
    ideal_dcg = dcg(ideal_relevance, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0

print(ndcg([3, 2, 3, 0, 1, 2], 6))