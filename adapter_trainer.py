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

    This dataset is designed to efficiently handle precomputed embeddings for triplet loss training.
    It takes precomputed embeddings for queries, positive samples (relevant documents), and negative samples (irrelevant documents).
    During training, it randomly selects a negative sample for each query and positive pair to form a triplet.

    Attributes:
        query_embs (torch.Tensor): Precomputed query embeddings. Shape: (num_queries, embedding_dimension).
        positive_embs (torch.Tensor): Precomputed positive embeddings. Shape: (num_queries, embedding_dimension).
        negative_embs (torch.Tensor): Precomputed negative embeddings. Shape: (num_negatives, embedding_dimension).
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
        """
        Return a triplet of embeddings (query, positive, negative) for a given index.

        A random negative embedding is selected for each query-positive pair.

        Args:
            idx (int): Index of the query-positive pair.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing the query embedding, positive embedding, and a randomly selected negative embedding.
        """
        # Randomly select a negative sample
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

    This scheduler linearly increases the learning rate from 0 to the initial learning rate during the warmup period
    and then linearly decreases it from the initial learning rate to 0 during the remaining training steps.

    Args:
        optimizer (Optimizer): Optimizer to wrap.
        num_warmup_steps (int): Number of warmup steps.
        num_training_steps (int): Total number of training steps.

    Returns:
        LambdaLR: Learning rate scheduler.
    """

    def lr_lambda(current_step: int) -> float:
        """
        Learning rate calculation function.

        Args:
            current_step (int): The current training step.

        Returns:
            float: The learning rate multiplier for the current step.
        """
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

    This class handles the training loop, evaluation, and saving of the LinearAdapter.
    It uses precomputed embeddings to speed up the training process.

    Attributes:
        device (torch.device): The device to run the training on (e.g., "cuda" or "cpu").
        base_model (SentenceTransformer): The base SentenceTransformer model.
        adapter (LinearAdapter): The linear adapter to be trained.
        train_query_embed (torch.Tensor): Precomputed embeddings for training queries.
        test_query_embed (torch.Tensor): Precomputed embeddings for test queries.
        train_chunk_embed (torch.Tensor): Precomputed embeddings for training chunks (positive samples).
        test_chunk_embed (torch.Tensor): Precomputed embeddings for test chunks (positive samples).
        negative_embed (torch.Tensor): Precomputed embeddings for negative samples.
        evaluator (ChromaEvaluator): Evaluator for assessing the adapter performance using ChromaDB.
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
            train_data (dict): Training data containing questions and chunks.  Must have keys "question" and "chunk".
            test_data (dict): Test data containing questions and chunks. Must have keys "question" and "chunk".
            negative_chunks (list[str]): List of negative chunks for triplet loss.
            device (str): The device to run the training on, defaults to "cuda".
        """
        self.device = torch.device(device)
        self.base_model = base_model.to(self.device)
        # Initialize the linear adapter with the embedding dimension of the base model
        # TODO: allow user to define this outside the fucntion
        self.adapter = MLPAdapter(
            self.base_model.get_sentence_embedding_dimension(), hidden_dim=2048).to(self.device)

        train_chunks: list[str] = train_data["chunk"].tolist()
        test_chunks: list[str] = test_data["chunk"].tolist()
        # Find unique chunks and their embeddings to avoid redundant computations
        unique_chunks: list[str] = list(set(train_chunks + test_chunks))
        print(f"Total unique chunks: {len(unique_chunks)}.")

        print("Pre-computing all embeddings...")
        print("This might take a while...")
        with torch.no_grad():
            # Encode the training and test queries
            self.train_query_embed = self.base_model.encode(
                train_data["question"].tolist(), convert_to_tensor=True, device=self.device)
            self.test_query_embed = self.base_model.encode(
                test_data["question"].tolist(), convert_to_tensor=True, device=self.device)

            unique_chunks_embed = self.base_model.encode(
                unique_chunks, convert_to_tensor=True, device=self.device)

            # Encode the negative chunks
            self.negative_embed = self.base_model.encode(
                negative_chunks, convert_to_tensor=True, device=self.device)

        # Get the indices of the train and test chunks in the unique chunks list
        train_indices = [unique_chunks.index(chunk) for chunk in train_chunks]
        test_indices = [unique_chunks.index(chunk) for chunk in test_chunks]

        # Use the indices to get the embeddings for the train and test chunks
        self.train_chunk_embed = unique_chunks_embed[train_indices]
        self.test_chunk_embed = unique_chunks_embed[test_indices]

        print("Initializing Chroma with precomputed embeddings...")
        chroma_client = chromadb.PersistentClient()
        # Create a Chroma collection with cosine distance metric
        collection = chroma_client.get_or_create_collection(
            name=str(hash(time.time())),
            metadata={"hnsw:space": "cosine"}
        )

        batch_size = 100
        # Batch insertion into Chroma for performance optimization
        # See https://docs.trychroma.com/production/administration/performance
        for i in tqdm(
            range(0, len(unique_chunks), batch_size),
            desc=f"Inserting chunks into Chroma with batch size {batch_size}",
        ):
            batch_end = min(i + batch_size, len(unique_chunks))
            batch_embeddings = unique_chunks_embed[i:batch_end].tolist()
            batch_documents = unique_chunks[i:batch_end]
            batch_ids = [f"id_{j}" for j in range(i, batch_end)]

            collection.add(
                ids=batch_ids, embeddings=batch_embeddings, documents=batch_documents)

        # Initialize the ChromaEvaluator for evaluating the adapter
        self.evaluator = ChromaEvaluator(
            collection, self.device, train_data["chunk"].tolist(), test_data["chunk"].tolist(), self.train_query_embed, self.test_query_embed)

        # Create the PrecomputedTripletDataset for training
        self.dataset = PrecomputedTripletDataset(
            self.train_query_embed, self.train_chunk_embed, self.negative_embed)

    def calculate_baseline(self) -> None:
        """
        Calculates and prints the baseline performance of the SentenceTransformer model
        without the adapter.  This is done by evaluating the model directly using the ChromaEvaluator.
        """
        result = self.evaluator.evaluate(None)
        print(f"Evaluation of baseline:")
        print(
            f"Train Set: Avg Hit Rate@10: {result['train']['average_hit_rate']:.4f}, Avg MRR@10: {result['train']['average_reciprocal_rank']:.4f}")
        print(
            f"Test Set: Avg Hit Rate@10: {result['test']['average_hit_rate']:.4f}, Avg MRR@10: {result['test']['average_reciprocal_rank']:.4f}")

    def train(
        self,
        num_epochs: int = 20,
        batch_size: int = 32,
        learning_rate: float = 3e-3,
        warmup_steps: int = 100,
        max_grad_norm: float = 1.0,
        margin: float = 1.0,
        save_path: str = None,
        eval_epoch: int = 5,
        save_epoch: int = 10
    ) -> None:
        """
        Fine-tunes the linear adapter.

        Args:
            num_epochs (int): Number of training epochs, defaults to 20.
            batch_size (int): Batch size for training, defaults to 32.
            learning_rate (float): Learning rate for the optimizer, defaults to 3e-3.
            warmup_steps (int): Number of warmup steps for the scheduler, defaults to 100.
            max_grad_norm (float): Maximum gradient norm for clipping, defaults to 1.0.
            margin (float): Margin for the TripletMarginLoss, defaults to 1.0.
            save_path (str): Path to save the adapter weights, defaults to None.
            eval_epoch (int): Evaluate every eval_epoch, defaults to 5.
            save_epoch (int): Save the adapter every save_epoch, defaults to 10.
        """
        print("Preparing for trianing...")
        # Create the DataLoader for training
        dataloader = DataLoader(
            self.dataset, batch_size=batch_size, shuffle=True)
        # Initialize the AdamW optimizer
        optimizer = AdamW(self.adapter.parameters(), lr=learning_rate)
        # Initialize the TripletMarginLoss
        triplet_loss = nn.TripletMarginLoss(margin=margin, p=2)
        # Calculate the total number of training steps
        total_steps = len(dataloader) * num_epochs
        # Create the learning rate scheduler with warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer, warmup_steps, total_steps)

        print("Started Training.")
        for epoch in range(num_epochs):
            self.adapter.train()
            total_loss = 0.0

            # Iterate over the batches in the DataLoader
            for batch in dataloader:
                # Move the batch tensors to the device
                q, p, n = [t.to(self.device, non_blocking=True) for t in batch]
                # Pass the query embeddings through the adapter
                adapted_q = self.adapter(q)
                # Calculate the triplet loss
                loss = triplet_loss(adapted_q, p, n)

                # Zero the gradients, perform backpropagation, and update the parameters
                optimizer.zero_grad()
                loss.backward()
                clip_grad_norm_(self.adapter.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()

                # Update the total loss
                total_loss += loss.item()

            print(
                f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}")

            if (epoch + 1) % eval_epoch == 0:
                # Evaluate the adapter
                eval_results = self.evaluator.evaluate(self.adapter)
                print(f"Evaluation after epoch {epoch+1}:")
                print(
                    f"Train Set: Avg Hit Rate@10: {eval_results['train']['average_hit_rate']:.4f}, Avg MRR@10: {eval_results['train']['average_reciprocal_rank']:.4f}")
                print(
                    f"Test Set: Avg Hit Rate@10: {eval_results['test']['average_hit_rate']:.4f}, Avg MRR@10: {eval_results['test']['average_reciprocal_rank']:.4f}")

            if save_path and (epoch + 1) % save_epoch == 0:
                # Save the adapter weights
                torch.save(self.adapter.state_dict(),
                           f"{save_path}/adapter_{epoch+1}.pth")
                print(
                    f"Adapter saved at {save_path}/adapter_{epoch+1}.pth")
        print("Training finished.")
