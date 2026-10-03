import numpy as np
from sentence_transformers import SentenceTransformer

# Gives 384-number embeddings, which is the size SCRLCache expects by default.
MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingModel:

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_embedding_dimension()

    # Embeddings are normalized here because the cache compares them with a
    # plain dot product, which is only cosine similarity for unit vectors.
    def embed_query(self, query: str) -> np.ndarray:
        emb = self.model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        return emb.astype(np.float32)

    def embed_documents(self, texts: list) -> list:
        embs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [e.astype(np.float32) for e in embs]


embedding_model = EmbeddingModel()
