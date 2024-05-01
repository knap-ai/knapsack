from functools import partial 
from typing import Generator

from llama_cpp import Llama

from knapsack import CFG, logger
from knapsack.models.ai_model import AIRequestParams
from knapsack.base.config import DeploymentType 
from knapsack.base.error import KnapsackException
from knapsack.managers.llm_model import LlmModelManager


class LlmGenerator(object):
    _instance = None
    llm: Llama | None = None
    model_name: str = ""
    stop: list[str] = ["</s>", "USER", "User:", "user:", "<|im_end|>", "[/INST]"]

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(LlmModelManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        self.setup_llm_model()

    def setup_llm_model(self):
        if self.llm is not None:
            return 
        # TODO: move the hard-coded stuff into the config file 
        # model_cfg = CFG.model
        mistral_v02_uri = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q5_K_M.gguf?download=true"
        gemma_2b_q5_uri = "https://huggingface.co/mlabonne/gemma-2b-GGUF/resolve/main/gemma-2b.Q5_K_M.gguf?download=true"
        gemma_2b_full_uri = "https://huggingface.co/google/gemma-2b/resolve/main/gemma-2b.gguf?download=true"

        model_uri = mistral_v02_uri 
        # model_uri = gemma_2b_q5_uri

        self.model_name = "Mistral-7B-Instruct-v0.2.Q5_K_M"
        # self.model_name = "Gemma-2B-Q5_K_M"

        try: 
            model_path = self._download_model(uri=mistral_v02_uri)
            # model_path = self._download_model(uri=model_uri)
        except Exception as e:
            raise KnapsackException(f"Couldn't download model: {model_uri}. Exception: {e}")

        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=16000,
            n_gpu_layers=200000,
            verbose=False,
            n_threads=6,
        )

    def gen(self, prompt: str, stream: bool = True):
        logger.debug(f"llm_complete - {self.model_name} - prompt: {prompt}")
        return self._llm_complete(prompt, stream)

    # def load_ai_model(self, ai_model: AIModel) -> bool:
    #     self.ai_model = ai_model
    #     self.llm_manager.load(ai_model)
    #     try:
    #         self.llm_manager.load(ai_model)
    #     #     self.llm: Llama = Llama(
    #     #         # model_path=str(ai_model.path), 
    #     #         # model_path=str(expanduser("~/Desktop/mistral-7b-instruct-v0.2.Q5_K_M.gguf")),
    #     #         model_path=str(expanduser("~/Desktop/mistral-7b-instruct-v0.1.q5_k_m.gguf")),
    #     #         # embedding=True,
    #     #         # n_ctx=ai_model.n_ctx,
    #     #         n_ctx=16000,
    #     #         n_gpu_layers=ai_model.n_gpu_layers,
    #     #         verbose=False,
    #     #         n_threads=ai_model.n_threads,
    #     #     )
    #     #     return True
    #     except Exception as e: 
    #         logger.exception(f"Exception loading AI model: {ai_model}; \n{e}")
    #         return False

    def _llm_complete(self, prompt: str, stream: bool = True) -> Generator | list[str]:
        if self.llm is None:
            raise KnapsackException(f"LLM not set.")

        full_prompt = "<s>[INST] Please respond to the questions and commands below. Please be succinct, " + \
            "but inclusive of all information that is helpful and fulfills the request.\n\n" + \
            f"{prompt}\n[/INST]" 

        full_prompt_num_tokens: int = len(self.llm.tokenize(full_prompt.encode()))

        if stream:
            return self._stream_llm_complete(prompt=full_prompt)
        else:
            max_tokens = 16000 - full_prompt_num_tokens - 50  # 50 is just a buffer to prevent errors. 
            print("PROCESSING VIA LLM")
            result = self.llm(full_prompt, max_tokens=max_tokens, stop=self.stop)
            print(f"ORIGINAL LLM RESULT: {result}")
            texts = [r["text"] for r in result.get("choices", [])]
            # create_timestamp = result["created"]
            # id = result["id"]
            return texts

    def _stream_llm_complete(self, prompt: str) -> Generator:
        ai = AIRequestParams(content=prompt)
        prompt_tokens: list[int] = self.llm.tokenize(prompt.encode())
        llm_generator = partial(
            self.llm.generate, top_k=ai.top_k, top_p=ai.top_p, temp=ai.temperature, 
            frequency_penalty=ai.frequency_penalty, presence_penalty=ai.presence_penalty, 
            tfs_z=ai.tfs_z, mirostat_mode=ai.mirostat, mirostat_tau=ai.mirostat_tau, 
            mirostat_eta=ai.mirostat_eta
        )
        try: 
            for token in llm_generator(tokens=prompt_tokens):
                try: 
                    words: str = self.llm.detokenize([token]).decode()
                    # print(words, end="")
                    if words == "" or not words or len(words) <= 0:
                        # continue
                        break 
                    yield words
                    if any(stop in words for stop in self.stop):
                        break
                except Exception as e:
                    logger.exception("Error detokenizing.")
        except RuntimeError as e:
            logger.exception("Runtime error generating tokens.")
