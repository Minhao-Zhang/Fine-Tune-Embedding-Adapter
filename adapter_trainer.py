from torch.optim import Optimizer
import random
import time
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import chromadb
from sentence_transformers import SentenceTransformer

from adapters import *
from evaluation import ChromaEvaluator


class PrecomputedTripletDataset(Dataset):
    """
    Dataset for precomputed triplet embeddings (query, positive, negative).

    Attributes:
        query_embs (torch.Tensor): Precomputed query embeddings.
        positive_embs (torch.Tensor): Precomputed positive embeddings.
        negative_embs (torch.Tensor): Precomputed negative embeddings.
    """

    def __init__(
        self,
        query_embs: torch.Tensor,
        positive_embs: torch.Tensor,
        negative_embs: torch.Tensor,
    ):
        """
        Initialize PrecomputedTripletDataset.

        Args:
            query_embs (torch.Tensor): Precomputed query embeddings.
            positive_embs (torch.Tensor): Precomputed positive embeddings.
            negative_embs (torch.Tensor): Precomputed negative embeddings.
        """
        self.query_embs = query_embs
        self.positive_embs = positive_embs
        self.negative_embs = negative_embs

    def __len__(self) -> int:
        """Return the number of query embeddings."""
        return len(self.query_embs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return a triplet of embeddings."""
        neg_idx = random.randint(0, len(self.negative_embs) - 1)
        return (
            self.query_embs[idx],
            self.positive_embs[idx],
            self.negative_embs[neg_idx],
        )


def get_linear_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int
) -> LambdaLR:
    """
    Create a linear learning rate scheduler with warmup.

    Args:
        optimizer (Optimizer): Optimizer to wrap.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Total number of training steps.

    Returns:
        LambdaLR: Learning rate scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda)


class AdapterTrainer:
    """
    Trainer class for fine-tuning a LinearAdapter on top of a SentenceTransformer model.

    Attributes:
        device (torch.device): The device to run the training on.
        base_model (SentenceTransformer): The base SentenceTransformer model.
        adapter (LinearAdapter): The linear adapter to be trained.
        train_query_embed (torch.Tensor): Precomputed embeddings for training queries.
        test_query_embed (torch.Tensor): Precomputed embeddings for test queries.
        train_chunk_embed (torch.Tensor): Precomputed embeddings for training chunks.
        test_chunk_embed (torch.Tensor): Precomputed embeddings for test chunks.
        negative_embed (torch.Tensor): Precomputed embeddings for negative samples.
        evaluator (ChromaEvaluator): Evaluator for assessing the adapter performance.
        dataset (PrecomputedTripletDataset): Dataset for training triplets.
    """

    def __init__(
        self,
        base_model: SentenceTransformer,
        train_data: dict,
        test_data: dict,
        negative_chunks: list[str],
        device: str = "cuda",
    ):
        """
        Initializes the AdapterTrainer.

        Args:
            base_model (SentenceTransformer): The base SentenceTransformer model.
            train_data (dict): Training data containing questions and chunks.
            test_data (dict): Test data containing questions and chunks.
            negative_chunks (list[str]): List of negative chunks for triplet loss.
            device (str): The device to run the training on, defaults to "cuda".
        """
        self.device = torch.device(device)
        self.base_model = base_model.to(self.device)
        self.adapter = LinearAdapter(
            self.base_model.get_sentence_embedding_dimension()).to(self.device)

        print("Pre-computing all embeddings...")
        print("This might take a while...")
        with torch.no_grad():
            self.train_query_embed = self.base_model.encode(
                train_data["question"].tolist(), convert_to_tensor=True, device=self.device)
            self.test_query_embed = self.base_model.encode(
                test_data["question"].tolist(), convert_to_tensor=True, device=self.device)
            self.train_chunk_embed = self.base_model.encode(
                train_data["chunk"].tolist(), convert_to_tensor=True, device=self.device)
            self.test_chunk_embed = self.base_model.encode(
                test_data["chunk"].tolist(), convert_to_tensor=True, device=self.device)
            self.negative_embed = self.base_model.encode(
                negative_chunks, convert_to_tensor=True, device=self.device)

        print("Initializing Chroma with precomputed embeddings...")
        chroma_client = chromadb.PersistentClient()
        collection = chroma_client.get_or_create_collection(
            name=str(hash(time.time())))

        all_chunks_embed = torch.cat(
            (self.train_chunk_embed, self.test_chunk_embed), dim=0)
        all_chunks = train_data["chunk"].tolist() + test_data["chunk"].tolist()

        batch_size = 100
        # Batch insertion into Chroma
        # Optimze based on https://docs.trychroma.com/production/administration/performance
        for i in tqdm(
            range(0, len(all_chunks), batch_size),
            desc=f"Inserting chunks into Chroma with batch size {batch_size}",
        ):
            batch_end = min(i + batch_size, len(all_chunks))
            batch_embeddings = all_chunks_embed[i:batch_end].tolist()
            batch_documents = all_chunks[i:batch_end]
            batch_ids = [f"id_{j}" for j in range(i, batch_end)]

            collection.add(
                ids=batch_ids, embeddings=batch_embeddings, documents=batch_documents)

        self.evaluator = ChromaEvaluator(collection, self.device, train_data["chunk"].tolist(
        ), test_data["chunk"].tolist(), self.train_query_embed, self.test_query_embed)

        self.dataset = PrecomputedTripletDataset(
            self.train_query_embed, self.train_chunk_embed, self.negative_embed)

    def calculate_baseline(self) -> None:
        """
        Calculates and prints the baseline performance of the SentenceTransformer model
        without the adapter.
        """
        result = self.evaluator.evaluate(None)
        print(f"Evaluation of baseline:")
        print(
            f"Train Set: Avg Hit Rate@10: {result['train']['average_hit_rate']:.4f}, Avg MRR@10: {result['train']['average_reciprocal_rank']:.4f}")
        print(
            f"Test Set: Avg Hit Rate@10: {result['test']['average_hit_rate']:.4f}, Avg MRR@10: {result['test']['average_reciprocal_rank']:.4f}")

    def train(
        self,
        num_epochs: int = 100,
        batch_size: int = 1024,
        learning_rate: float = 3e-3,
        warmup_steps: int = 5,
        max_grad_norm: float = 1.0,
        margin: float = 1.0,
        save_path: str = None,
    ) -> None:
        """
        Fine-tunes the linear adapter.

        Args:
            num_epochs (int): Number of training epochs, defaults to 100.
            batch_size (int): Batch size for training, defaults to 1024.
            learning_rate (float): Learning rate for the optimizer, defaults to 3e-3.
            warmup_steps (int): Number of warmup steps for the scheduler, defaults to 5.
            max_grad_norm (float): Maximum gradient norm for clipping, defaults to 1.0.
            margin (float): Margin for the TripletMarginLoss, defaults to 1.0.
            save_path (str): Path to save the adapter weights, defaults to None.
        """
        print("Preparing for trianing...")
        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.adapter.parameters(), lr=learning_rate)
        triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
        total_steps = len(dataloader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps)

        print("Started Training.")
        for epoch in range(num_epochs):
            self.adapter.train()
            total_loss = 0.0

            for batch in dataloader:
                q, p, n = [t.to(self.device, non_blocking=True) for t in batch]
                adapted_q = self.adapter(q)
                loss = triplet_loss(adapted_q, p, n)

                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.adapter.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

            if (epoch + 1) % 5 == 0:
                eval_results = self.evaluator.evaluate(self.adapter)
                print(f"Evaluation after epoch {epoch+1}:")
                print(
                    f"Train Set: Avg Hit Rate@10: {eval_results['train']['average_hit_rate']:.4f}, Avg MRR@10: {eval_results['train']['average_reciprocal_rank']:.4f}")
                print(
                    f"Test Set: Avg Hit Rate@10: {eval_results['test']['average_hit_rate']:.4f}, Avg MRR@10: {eval_results['test']['average_reciprocal_rank']:.4f}")

            if save_path and (epoch + 1) % 10 == 0:
                torch.save(self.adapter.state_dict(),
                           f"{save_path}/adapter_{epoch+1}.pth")
                print(
                    f"Adapter saved at {save_path}/adapter_{epoch+1}.pth")
        print("Training finished.")
