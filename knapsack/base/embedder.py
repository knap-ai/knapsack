from llama_cpp import Llama

from knapsack import CFG, logger
from knapsack.base.error import KnapsackException
from knapsack.memory.embed import create_embeddings


class Embedder(object):
    _instance = None
    embedding_model: Llama | None = None
    model_name: str = ""

    def __init__(self):
        self.setup_embed_model()

    def setup_embed_model(self):
        if self.embedding_model is not None:
            return 
        model_cfg = CFG.embedding_model
        self.embedding_model = Llama(
            model_path=str(model_cfg.llama_cpp.location), 
            embedding=True,
            n_gpu_layers=model_cfg.llama_cpp.n_gpu_layers,
        )
        logger.debug(f"LLAMA EMBEDDING MODEL LOADED")

    def embed(self, data: str | list[str]):
        if isinstance(data, str):
            data = [data]

        logger.debug(f"embedding with {self.model_name}")
        return create_embeddings(self.embedding_model, data)
