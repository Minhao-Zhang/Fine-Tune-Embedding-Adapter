from chromadb.utils.embedding_functions import sentence_transformer_embedding_function
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
import torch
from typing import cast

from adapters import BaseAdapter


class MyEmbeddingFunction(EmbeddingFunction):
    def __init__(self,
                 model_name_or_path,
                 adpater_file,
                 device="cuda",
                 normalize_embeddings=False,
                 half=False):
        super().__init__()
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(
            model_name_or_path=model_name_or_path, device=device)
        self.model.encode()
        if half:
            self.model.half()
        self.adpter: BaseAdapter = torch.load(adpater_file)

    def __call__(self, input: Documents) -> Embeddings:
        return cast(
            Embeddings,
            [
                self.adapter(embedding).cpu().detach().numpy()
                for embedding in self._model.encode(
                    list(input),
                    convert_to_tensor=True,
                    normalize_embeddings=self.normalize_embeddings,
                )
            ],
        )
        return embeddings
