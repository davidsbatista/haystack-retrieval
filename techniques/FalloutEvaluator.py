from typing import Any, Dict, List
from haystack import component

@component
class FalloutEvaluator:
    """
    Evaluator that calculates Fallout score (false positive rate among retrieved documents).
    """

    @staticmethod
    def _compute_fallout(ground_truth_documents: List[str], retrieved_documents: List[str]) -> float:
        """Compute fallout for a single set of documents."""
        ground_truth_set = set(ground_truth_documents)
        retrieved_set = set(retrieved_documents)
        non_relevant = retrieved_set - ground_truth_set
        
        return len(non_relevant) / len(retrieved_set) if retrieved_set else 0.0

    @component.output_types(
        fallout_scores=List[float],
        mean_fallout=float
    )
    def run(self, ground_truth_documents: List[List[str]], retrieved_documents: List[List[str]]) -> Dict[str, Any]:
        fallout_scores = []

        for ground_truth, retrieved in zip(ground_truth_documents, retrieved_documents):
            fallout = self._compute_fallout(ground_truth, retrieved)
            fallout_scores.append(fallout)

        return {
            "individual_scores": fallout_scores,
            "score": sum(fallout_scores) / len(fallout_scores) if fallout_scores else 0.0
        } 