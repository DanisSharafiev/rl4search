from src.search.candidate_generator import CandidateGenerator

class ContextualBanditReranker:
    def __init__(self, candidates):
        self.candidates = candidates

    def reward_function(self, query, document):
        # TODO: Implement a proper reward function based on relevance
        # Linear function
        return 1.0
    
    def rerank(self, query):
        ranked_candidates = sorted(self.candidates, key=lambda x: self.reward_function(query, x), reverse=True)
        return ranked_candidates
    