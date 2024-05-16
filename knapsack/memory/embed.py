import typing as t

import numpy as np
import pyarrow as pa
import requests
from llama_cpp import Llama
from sentence_transformers import SentenceTransformer

from knapsack.base.util import procure_cfg


def create_embedding(
    llm: t.Union[Llama, SentenceTransformer], 
    content: str
) -> list[float]:
    if isinstance(llm, Llama):
        result = llm.create_embedding(content)
        if 'data' in result: 
            if 'embedding' in result['data'][0]:
                return result['data'][0]['embedding']
    elif isinstance(llm, SentenceTransformer):
        embedding = llm.encode(content)
        return list(embedding)
    return []


def create_embeddings(
    llm: t.Union[Llama, SentenceTransformer], 
    content: list[str],
) -> list[list[float]]:
    if isinstance(llm, Llama):
        result = llm.create_embedding(content)
        if 'data' in result: 
            data = result['data']
            return [data[i]['embedding'] for i, _ in enumerate(data)]
    elif isinstance(llm, SentenceTransformer):
        embeddings = llm.encode(content)
        return list(embeddings)
    return []


def normalize_vectors(vectors, ndim):
      if ndim is None:
          ndim = len(next(iter(vectors)))
      values = np.array(vectors, dtype="float32").ravel()
      return pa.FixedSizeListArray.from_arrays(values, list_size=ndim)
