import numpy as np
import torch
from typing import List
from chromadb import Collection


def reciprocal_rank(retrieved_docs: List[str], ground_truth: str, k: int) -> float:
    """
    Compute the reciprocal rank metric for a single query.

    Args:
        retrieved_docs (List[str]): List of retrieved document identifiers.
        ground_truth (str): The correct document identifier.
        k (int): The maximum rank to consider.

    Returns:
        float: Reciprocal rank score (0 if not found within k).
    """
    try:
        rank = retrieved_docs.index(ground_truth) + 1
        return 1.0 / rank if rank <= k else 0.0
    except ValueError:
        return 0.0


def hit_rate(retrieved_docs: List[str], ground_truth: str, k: int) -> float:
    """
    Compute the hit rate metric for a single query.

    Args:
        retrieved_docs (List[str]): List of retrieved document identifiers.
        ground_truth (str): The correct document identifier.
        k (int): The maximum rank to consider.

    Returns:
        float: 1.0 if the ground truth is in the top k retrieved docs, else 0.0.
    """
    return 1.0 if ground_truth in retrieved_docs[:k] else 0.0


class ChromaEvaluator:
    """
    Evaluates the embedding adapter using a ChromaDB collection.

    Attributes:
        collection (chromadb.Collection): Collection used for queries.
        train_ground_truths (List[str]): List of ground truth documents for training.
        test_ground_truths (List[str]): List of ground truth documents for testing.
        device (torch.device): Device for computations.
        train_question_embs (torch.Tensor): Precomputed training question embeddings.
        test_question_embs (torch.Tensor): Precomputed testing question embeddings.
    """

    def __init__(
        self,
        collection: Collection,
        device: torch.device,
        train_ground_truths: List[str],
        test_ground_truths: List[str],
        train_question_embs: torch.Tensor,
        test_question_embs: torch.Tensor,
    ):
        """
        Initialize ChromaEvaluator.

        Args:
            collection (chromadb.Collection): ChromaDB collection.
            device (torch.device): Computation device.
            train_ground_truths (List[str]): Ground truths for training queries.
            test_ground_truths (List[str]): Ground truths for test queries.
            train_question_embs (torch.Tensor): Training question embeddings.
            test_question_embs (torch.Tensor): Testing question embeddings.
        """
        self.collection = collection
        self.device = device
        self.train_ground_truths = train_ground_truths
        self.test_ground_truths = test_ground_truths
        self.train_question_embs = train_question_embs
        self.test_question_embs = test_question_embs

    def evaluate(
        self, adapter: torch.nn.Module, k: int = 10, batch_size: int = 1024
    ) -> dict:
        """
        Evaluate adapter performance on train and test splits.

        Args:
            adapter (torch.nn.Module): The adapter model.
            k (int, optional): Number of top documents to consider. Defaults to 10.
            batch_size (int, optional): Batch size for evaluation. Defaults to 1024.

        Returns:
            dict: Evaluation metrics for train and test splits.
        """
        if adapter is not None:
            adapter.eval()
        train_metrics = self._evaluate_split(
            adapter, self.train_question_embs, self.train_ground_truths, k, batch_size
        )
        test_metrics = self._evaluate_split(
            adapter, self.test_question_embs, self.test_ground_truths, k, batch_size
        )
        return {"train": train_metrics, "test": test_metrics}

    def _evaluate_split(
        self,
        adapter: torch.nn.Module,
        question_embs: torch.Tensor,
        ground_truths: List[str],
        k: int,
        batch_size: int,
    ) -> dict:
        adapted_embs_list = None
        if adapter is not None:
            adapted_batches = []
            with torch.no_grad():
                for i in range(0, len(question_embs), batch_size):
                    batch = question_embs[i: i + batch_size].to(self.device)
                    adapted_batches.append(adapter(batch).cpu())
            adapted_embs = torch.cat(adapted_batches).numpy()
            adapted_embs_list = adapted_embs.tolist()  # Use adapted embeddings for query
            query_embs_list = adapted_embs_list
        else:
            # Use base model embeddings for query
            query_embs_list = question_embs.cpu().numpy().tolist()

        # Batch query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embs_list, n_results=k, include=[
                "documents"]
        )

        hit_rates = []
        reciprocal_ranks = []
        for idx, docs in enumerate(results["documents"]):
            hr = hit_rate(docs, ground_truths[idx], k)
            hit_rates.append(hr)
            reciprocal_ranks.append(
                reciprocal_rank(docs, ground_truths[idx], k))
        return {
            "average_hit_rate": np.mean(hit_rates),
            "average_reciprocal_rank": np.mean(reciprocal_ranks),
        }
