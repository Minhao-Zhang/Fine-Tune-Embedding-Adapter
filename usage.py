from chromadb.utils.embedding_functions import sentence_transformer_embedding_function
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
import torch
from typing import cast

from adapters import BaseAdapter


class MyEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function that uses SentenceTransformer and adapter.
    """

    def __init__(
        self,
        model_name_or_path,
        adpater_file,
        device="cuda",
        normalize_embeddings=False,
        half=False,
    ):
        """
        Initialize MyEmbeddingFunction.

        Args:
            model_name_or_path (str): Path to pretrained model or model identifier from huggingface.co.
            adpater_file (str): Path to the adapter file.
            device (str, optional): Device to use for computation. Defaults to "cuda".
            normalize_embeddings (bool, optional): Whether to normalize embeddings. Defaults to False.
            half (bool, optional): Whether to use half precision. Defaults to False.
        """
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        # Load sentence transformer model
        self.model = SentenceTransformer(
            model_name_or_path=model_name_or_path, device=device)
        self.model.encode(["TEST"])  # encode once to load model into memory
        if half:
            self.model.half()  # use half precision
        # Load adapter
        self.adpter: BaseAdapter = torch.load(adpater_file)

    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for input documents.

        Args:
            input (Documents): List of documents to embed.

        Returns:
            Embeddings: List of embeddings.
        """
        # Generate embeddings using sentence transformer model and apply adapter
        embeddings = cast(
            Embeddings,
            [
                self.adpter(self.model.encode(sentences=[
                            sentence], convert_to_tensor=True, normalize_embeddings=self.normalize_embeddings)[0]).cpu().detach().numpy()
                for sentence in input
            ],
        )
        return embeddings
