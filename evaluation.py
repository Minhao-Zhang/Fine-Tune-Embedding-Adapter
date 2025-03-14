import numpy as np
import torch
from typing import List
from chromadb import Collection


def reciprocal_rank(retrieved_docs: List[str], ground_truth: str, k: int) -> float:
    try:
        rank = retrieved_docs.index(ground_truth) + 1
        return 1.0 / rank if rank <= k else 0.0
    except ValueError:
        return 0.0


def hit_rate(retrieved_docs: List[str], ground_truth: str, k: int) -> float:
    return 1.0 if ground_truth in retrieved_docs[:k] else 0.0


def precision_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    retrieved_at_k = retrieved_docs[:k]
    return sum(1 for doc in retrieved_at_k if doc in relevant_docs) / k


def recall_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    return sum(1 for doc in retrieved_docs[:k] if doc in relevant_docs) / len(relevant_docs)


def average_precision(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    score, num_hits = 0.0, 0
    for i, doc in enumerate(retrieved_docs[:k]):
        if doc in relevant_docs:
            num_hits += 1
            score += num_hits / (i + 1)
    return score / len(relevant_docs) if relevant_docs else 0.0


def ndcg_at_k(retrieved_docs: List[str], relevant_docs: List[str], k: int) -> float:
    def dcg(scores):
        return sum((2**s - 1) / np.log2(i + 2) for i, s in enumerate(scores))

    ideal_scores = [1] * min(k, len(relevant_docs)) + \
        [0] * (k - min(k, len(relevant_docs)))
    actual_scores = [
        1 if doc in relevant_docs else 0 for doc in retrieved_docs[:k]]

    return dcg(actual_scores) / dcg(ideal_scores) if dcg(ideal_scores) > 0 else 0.0


class ChromaEvaluator:
    def __init__(
        self,
        collection: Collection,
        device: torch.device,
        train_ground_truths: List[str],
        test_ground_truths: List[str],
        train_question_embs: torch.Tensor,
        test_question_embs: torch.Tensor,
    ):
        self.collection = collection
        self.device = device
        self.train_ground_truths = train_ground_truths
        self.test_ground_truths = test_ground_truths
        self.train_question_embs = train_question_embs
        self.test_question_embs = test_question_embs

    def evaluate(self, adapter: torch.nn.Module, k: int = 10, batch_size: int = 1024) -> dict:
        if adapter is not None:
            adapter.eval()
        train_metrics = self._evaluate_split(
            adapter, self.train_question_embs, self.train_ground_truths, k, batch_size)
        test_metrics = self._evaluate_split(
            adapter, self.test_question_embs, self.test_ground_truths, k, batch_size)
        return {"train": train_metrics, "test": test_metrics}

    def _evaluate_split(self, adapter: torch.nn.Module, question_embs: torch.Tensor, ground_truths: List[str], k: int, batch_size: int) -> dict:
        adapted_embs_list = None
        if adapter is not None:
            adapted_batches = []
            with torch.no_grad():
                for i in range(0, len(question_embs), batch_size):
                    batch = question_embs[i: i + batch_size].to(self.device)
                    adapted_batches.append(adapter(batch).cpu())
            adapted_embs = torch.cat(adapted_batches).numpy()
            adapted_embs_list = adapted_embs.tolist()
            query_embs_list = adapted_embs_list
        else:
            query_embs_list = question_embs.cpu().numpy().tolist()

        results = self.collection.query(
            query_embeddings=query_embs_list, n_results=k, include=["documents"])

        hit_rates, reciprocal_ranks, precisions, recalls, ndcgs, aps = [], [], [], [], [], []
        for idx, docs in enumerate(results["documents"]):
            # Modify this if multiple relevant docs exist
            relevant_docs = [ground_truths[idx]]
            hit_rates.append(hit_rate(docs, ground_truths[idx], k))
            reciprocal_ranks.append(
                reciprocal_rank(docs, ground_truths[idx], k))
            precisions.append(precision_at_k(docs, relevant_docs, k))
            recalls.append(recall_at_k(docs, relevant_docs, k))
            ndcgs.append(ndcg_at_k(docs, relevant_docs, k))
            aps.append(average_precision(docs, relevant_docs, k))

        return {
            "average_hit_rate": np.mean(hit_rates),
            "average_reciprocal_rank": np.mean(reciprocal_ranks),
            "average_precision": np.mean(precisions),
            "average_recall": np.mean(recalls),
            "average_ndcg": np.mean(ndcgs),
            "mean_average_precision": np.mean(aps),
        }
