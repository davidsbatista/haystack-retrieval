from typing import Any, Dict, List
from haystack import component

@component
class RecallEvaluator:
    """
    Evaluator that calculates Recall score (ratio of retrieved relevant documents among all relevant documents).
    """

    @staticmethod
    def _compute_recall(ground_truth_documents: List[str], retrieved_documents: List[str]) -> float:
        """Compute recall for a single set of documents."""
        ground_truth_set = set(ground_truth_documents)
        retrieved_set = set(retrieved_documents)
        relevant_and_retrieved = ground_truth_set.intersection(retrieved_set)
        
        return len(relevant_and_retrieved) / len(ground_truth_set) if ground_truth_set else 0.0

    @component.output_types(
        recall_scores=List[float],
        mean_recall=float
    )
    def run(self, ground_truth_documents: List[List[str]], retrieved_documents: List[List[str]]) -> Dict[str, Any]:
        recall_scores = []

        for ground_truth, retrieved in zip(ground_truth_documents, retrieved_documents):
            recall = self._compute_recall(ground_truth, retrieved)
            recall_scores.append(recall)

        return {
            "individual_scores": recall_scores,
            "score": sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
        }