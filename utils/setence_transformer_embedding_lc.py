import logging
from typing import List

# Configure Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from chromadb.utils.embedding_functions.sentence_transformer_embedding_function import SentenceTransformerEmbeddingFunction
from langchain.embeddings.base import Embeddings

import numpy as np

def convert_ndarray_to_float(data: List[List[np.ndarray]]) -> List[List[float]]:
    """
    Converts a nested list of NumPy ndarrays to a nested list of Python floats.

    Args:
        data: A nested list where each element is a NumPy ndarray containing integers or floats.

    Returns:
        A nested list where each element is a Python float.
    """
    return [
        [float(item) for array in sublist for item in array.tolist()]
        for sublist in data
    ]

class ChromaSentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return convert_ndarray_to_float([self.embedding_function([text]) for text in texts])


    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


# Example usage
if __name__ == "__main__":
    embedding_model = ChromaSentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
