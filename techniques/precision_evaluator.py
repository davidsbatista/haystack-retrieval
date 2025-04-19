from typing import Any, Dict, List
from haystack import component

@component
class PrecisionEvaluator:
    """
    Evaluator that calculates Precision score (ratio of relevant documents among retrieved documents).
    """

    @staticmethod
    def _compute_precision(ground_truth_documents: List[str], retrieved_documents: List[str]) -> float:
        """Compute precision for a single set of documents."""
        ground_truth_set = set(ground_truth_documents)
        retrieved_set = set(retrieved_documents)
        relevant_and_retrieved = ground_truth_set.intersection(retrieved_set)
        
        return len(relevant_and_retrieved) / len(retrieved_set) if retrieved_set else 0.0

    @component.output_types(
        precision_scores=List[float],
        mean_precision=float
    )
    def run(self, ground_truth_documents: List[List[str]], retrieved_documents: List[List[str]]) -> Dict[str, Any]:
        precision_scores = []

        for ground_truth, retrieved in zip(ground_truth_documents, retrieved_documents):
            precision = self._compute_precision(ground_truth, retrieved)
            precision_scores.append(precision)

        return {
            "individual_scores": precision_scores,
            "score": sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
        } 