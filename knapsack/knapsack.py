import asyncio
import schedule
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from os import environ, makedirs
from os.path import expanduser
from pathlib import Path
from threading import Lock
from tqdm import tqdm
from typing import Any

import numpy as np
from cachetools import cached, LRUCache
from sentence_transformers import SentenceTransformer
from unstructured.partition.auto import partition

from knapsack import CFG, logger
from knapsack.base.cache import cache_custom_key, clear_cache_for_collection
from knapsack.base.error import KnapsackException
from knapsack.base.util import Timer, cosine_similarity
from knapsack.connectors.arxiv import ArXivConnector
from knapsack.connectors.base_connector import BaseConnector
from knapsack.connectors.bioarxiv import BioArXivConnector
from knapsack.knap_document import KnapDocument
from knapsack.memory.qdrant import QdrantVectorSearcher
from knapsack.util.upsert_stats import UpsertStats


class Knapsack:
    lru_cache = LRUCache(maxsize=1024)
    cache_lock: Lock = Lock()

    def __new__(cls):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Knapsack, cls).__new__(cls)
        return cls.instance

    def __init__(self):
        environ['PYTHONHASHSEED'] = "0"
        self.connectors = [self.initialize_connector(connector) for connector in CFG.connectors]

        self._dir: Path = Path(expanduser(CFG.knapsack_dir))
        makedirs(str(self._dir), exist_ok=True)

        self.ks_backup_dir: Path = self._dir / Path("snapshots")
        makedirs(str(self.ks_backup_dir), exist_ok=True)        

        self._log_dir: Path = Path(expanduser(CFG.log_dir))
        makedirs(str(self._log_dir), exist_ok=True)

        self.embedder = SentenceTransformer(CFG.embedder.id)

        self.vector_searcher: QdrantVectorSearcher = QdrantVectorSearcher(
            qdrant_home=self._dir / Path("qdrant"),
            embedder=self.embedder,
            dimensions=CFG.embedder.size,
            ks_backup_dir=self.ks_backup_dir,
        )

        self.timer = Timer()

    def embed(self, content: list[str]):
        return self.embedder.embed(content)

    def initialize_connector(self, connector_config):
        connector_name = connector_config['name']
        params = connector_config.copy()
        del params['name']
        if connector_name == "ArXivConnector":
            return ArXivConnector(**params)
        elif connector_name == "BioArXivConnector":
            return BioArXivConnector(**params)
        else:
            raise ValueError(f"Unsupported connector: {connector_name}")

    def run(self):
        print(f"fetching...")
        for connector in tqdm(self.connectors, desc="Running connectors"):
            self._learn_from_connector(connector)  
        # for connector in tqdm(self.connectors, desc="Running connectors"):
        #     schedule.every(connector.interval).minutes.do(self._learn_from_connector(connector))
        # 
        # while True:
        #     schedule.run_pending()
        #     time.sleep(1)

    def _learn_from_connector(self, conn: BaseConnector):
        for results in conn.fetch():
            tags = conn.knapsack_tags()
            data = [r.to_dict() for r in results]
            ids = [r.uuid() for r in results]
            upsert_stats = self.learn(
                data=data,
                collection="knapsack",
                tags=tags,
                ids=ids,
                wait=True,
            )
            print(f"Upsert stats: {upsert_stats}")

    def learn(
        self, 
        data: list[dict[str, Any]] | str | Path, 
        collection: str, 
        tags: dict[str, Any],
        ids: list[str],
        wait: bool = False
    ) -> UpsertStats | None:
        """
        data - either data itself, or a uri that points to data. 
        collection - specifies the collection of vector embeddings where all
            new learnings are to be stored.
        tags - metadata about data, or, in the case of a .csv, the columns
            that should be embedded.
        ids - list of user-supplied IDs for the individual data points. Can be 'null'
            if the data has a 'uuid' field (or column in the .csv)
        """
        # logger.debug(f"LEARN: data:{data} \ncollection:{collection}\ntags:{tags}\nids:{ids}\nwait:{wait}")
        doc = KnapDocument(data=data, tags=tags)
        result = None
        # logger.debug(f"Cache keys before clear: {self.lru_cache.keys()}")
        clear_cache_for_collection(self.lru_cache, self.cache_lock, collection_to_clear=collection)
        # logger.debug(f"Cache keys after clear: {self.lru_cache.keys()}")
        if wait: 
            # logger.debug(f"about to LEARN: ids {ids}\n collection {collection}")
            result = asyncio.run(self.vector_searcher.commit(
                doc=doc, collection=collection, uuids=ids,
            ))
            logger.debug(f"LEAVING LEARN: {collection}")
        else:
            asyncio.run(self.nonblocking_commit(doc=doc, collection=collection, ids=ids))

        return result 

    async def nonblocking_commit(
        self, 
        doc: KnapDocument, 
        collection: str, 
        ids: list[str] | list[int],
    ) -> None:
        task = asyncio.create_task(self.nonblocking_commit_helper(doc=doc, collection=collection, ids=ids))

    async def nonblocking_commit_helper(
        self, 
        doc: KnapDocument, 
        collection: str, 
        ids: list[str] | list[int],
    ) -> None:
        uuid_iter = doc.uuids()
        uuids = []
        for u in uuid_iter:
            uuids += u
        asyncio.run(self.vector_searcher.commit(doc=doc, collection=collection, uuids=uuids))

    @cached(cache=lru_cache, key=partial(cache_custom_key, 'semantic_search'), lock=cache_lock)
    def semantic_search(
        self,
        query: str,
        collection: str, 
        num_results: int,
        # return_columns: list[str] = ['uuid'],
        filter: str | None = None,
        with_vector: bool = False,
    ) -> list[dict[str, Any]]:
        self.timer.start('ss - total')
        query_embedding: list[float] = self.embedder.embed(query)[0]
        results = asyncio.run(self.vector_searcher.find_nearest_from_array(
            query_embedding, 
            collection=collection,
            n=num_results, 
            filter=filter, 
            return_original=True,
            with_vector=with_vector,
        ))
        self.timer.end('ss - total')
        return results

    @cached(cache=lru_cache, key=partial(cache_custom_key, 'cross_retrieve'), lock=cache_lock)
    def cross_retrieve(
        self,
        uuids: list[str],
        collection_src: str, 
        collection_target: str, 
        num_results: int = 20,
        return_original: bool = False,
        filter: str | None = None,
        with_vector: bool = False,
    ) -> dict[str, list[dict[str, Any]]]:
        # TODO: what happens when multiple ids are passed, but some of them don't return 
        # any results? The list should still be len(ids), but it looks like it isn'
        # logger.debug(f"cr id_layer: {id_layer}")
        # logger.debug(f"cr uuids: {uuids}")
        vectors = self.find_by(uuids, collection=collection_src, with_vector=True)
        # logger.debug(f"cr len vectors: {len(vectors)}")
        vec_search_results = {}
        for uuid, v in vectors.items():
            vec_search_results[uuid] = self.vector_search(
                    vector=v['vector'], 
                    collection=collection_target, 
                    num_results=num_results, 
                    return_original=return_original, 
                    filter=filter,
                    with_vector=with_vector,
            )
        return vec_search_results

    @cached(cache=lru_cache, key=partial(cache_custom_key, 'retrieve'), lock=cache_lock)
    def retrieve(
        self,
        uuids: list[str],
        collection: str, 
        with_vector: bool = False,
    ):
        results = asyncio.run(self.vector_searcher.find_by_uuid(
            uuids=uuids, collection=collection, with_vector=with_vector,
        ))
        return results 

    def vector_search(
        self,
        vector: list[float],
        collection: str, 
        num_results: int = 20,
        return_original: bool = False,
        filter: str | None = None,
        with_vector: bool = False,
    ) -> list[dict[str, Any]]:
        results = asyncio.run(self.vector_searcher.find_nearest_from_array(
            h=np.array(vector), 
            collection=collection, 
            n=num_results, 
            filter=filter,
            return_original=return_original,
            with_vector=with_vector,
        ))
        return results

    def vector_compare(
        self,
        uuids1: list[str], 
        collection1: str, 
        uuids2: list[str], 
        collection2: str, 
        num_results: int = 20,
        return_original: bool = False,
        with_payload: bool = True,
        with_vector: bool = False,
    ) -> list[dict[str, Any]]:
        vectors1 = self.find_by(uuids1, collection=collection1, with_payload=with_payload, with_vector=True)
        vectors2 = self.find_by(uuids2, collection=collection2, with_payload=with_payload, with_vector=True)
        results = []
        for id1, v1 in vectors1.items():
            for id2, v2 in vectors2.items():
                result = {}
                result['score'] = cosine_similarity(np.array(v1['vector']), np.array(v2['vector']))
                result['payload1'] = v1.get('payload', {})
                result['payload2'] = v2.get('payload', {})
                result['uuid1'] = id1
                result['uuid2'] = id2
                if with_vector:
                    result['vector1'] = v1['vector']
                    result['vector2'] = v2['vector']
                results.append(result)
        return results

    def vector_list(
        self,
        collection: str, 
        num_results: int = 200,
        with_payload: bool = True,
        filter: str | None = None,
        page: str | None = None,
    ):
        result = asyncio.run(self.vector_searcher.list_vectors(
            collection=collection, 
            num_results=num_results, 
            with_payload=with_payload, 
            filter=filter,
            page=page,
        ))
        return result

    @cached(cache=lru_cache, key=partial(cache_custom_key, 'find_by'), lock=cache_lock)
    def find_by(
        self,
        uuids: list[str],
        collection: str,
        with_vector: bool = False,
        with_payload: bool = True,
    ):
        return asyncio.run(self.vector_searcher.find_by_uuid(
            uuids=uuids,
            collection=collection,
            with_vector=with_vector,
            with_payload=with_payload,
        ))

    def delete(
        self,
        uuids: list[str],
        collection: str,
    ):
        clear_cache_for_collection(self.lru_cache, self.cache_lock, collection_to_clear=collection)
        return asyncio.run(self.vector_searcher.delete(
            uuids=uuids,
            collection=collection,
        ))

    def backup(
        self,
        name: str,
        collection: str,
    ) -> Path | None:
        """
        Return Path of successful backup, None otherwise.
        """
        ks_backup_path: Path = self.db.backup(name)
        is_success = asyncio.run(self.vector_searcher.backup(
            name=name, collection=collection
        ))
        if is_success:
            return ks_backup_path
        return None

    def restore(
        self,
        collection: str,
        ks_backup_path: Path,
    ):
        """
        Return True for success, false otherwise.
        """
        if not ks_backup_path.exists():
            raise KnapsackException(f"can't find backup: {ks_backup_path}")
        self.db.restore(ks_backup_path)
        return asyncio.run(self.vector_searcher.restore(
            collection=collection, ks_backup_path=ks_backup_path, 
        ))

    def list_snapshots(self, collection: str) -> list[str]:
        snapshots = asyncio.run(
                self.vector_searcher.list_snapshots(collection=collection)
        )
        return snapshots

    def list_collections(self) -> list[str]:
        return asyncio.run(self.vector_searcher.list_collections())

    def copy_collection(self, existing_collection: str, new_collection: str) -> bool:
        try:
            asyncio.run(self.vector_searcher.copy_collection(existing_collection, new_collection))
        except Exception:
            logger.exception("Error in copy_collection.")
            return False
        return True

    def delete_collection(self, collection: str) -> bool:
        try:
            asyncio.run(self.vector_searcher.delete_collection(collection))
            clear_cache_for_collection(self.lru_cache, self.cache_lock, collection_to_clear=collection)
        except Exception:
            logger.exception("Error in delete_collection.")
            return False
        return True

    def count_collection(self, collection: str) -> int:
        return asyncio.run(self.vector_searcher.count_collection(collection))

    def collection_info(self, collection: str):
        return asyncio.run(self.vector_searcher.collection_info(collection))

    # def llm_complete(self, prompt: str, stream: bool = True):
    #     return self.llm_generator.gen(prompt, False)

    def extract_text_from_file(self, source: str | Path) -> str:
        if isinstance(source, str) or isinstance(source, Path):
            src_path = Path(source)
        else: 
            raise KnapsackException(f"extract_text_from_file expects either str or Path.")

        if not src_path.exists():
            raise KnapsackException(f"File at {src_path} does not exis")

        # TODO: house this functionality in KnapDocumen
        elements = partition(str(src_path))
        # TODO: the metadata could help produce better 
        # doc text extractions.
        # for el in elements: 
        #     print(el.metadata)
        extracted_text = "\n\n".join([str(el) for el in elements])
        return extracted_text

