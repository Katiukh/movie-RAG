"""Corpus-grounded retrieval evaluation, independent of search backends."""

from movie_rag.evaluation.dataset import load_eval_dataset, validate_eval_dataset
from movie_rag.evaluation.metrics import recall_at_k
from movie_rag.evaluation.runner import evaluate_retriever

__all__ = [
    "evaluate_retriever",
    "load_eval_dataset",
    "recall_at_k",
    "validate_eval_dataset",
]
