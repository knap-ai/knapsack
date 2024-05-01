import os
import shutil
import typing as t
from datetime import datetime
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd
import pyarrow as pa
import requests
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.http.models import (
    Batch, 
    CollectionsResponse,
    Distance, 
    InitFrom,
    PointIdsList, 
    Record,
    SnapshotDescription,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

import knapsack.base.util as util
from knapsack import CFG, logger
from knapsack.base.error import KnapsackException
from knapsack.base.util import Timer
from knapsack.knap_document import KnapDocument
from knapsack.memory.embed import create_embeddings
from knapsack.util.sql_parser import convert_sql_where_to_qdrant_filter
from knapsack.util.upsert_stats import UpsertStats


import nest_asyncio
nest_asyncio.apply()


class QdrantVectorSearcher(object):

    def __init__(
        self,
        qdrant_home: Path,
        embedder: SentenceTransformer,
        dimensions: int,
        ks_backup_dir: Path,
        h: t.Optional[np.ndarray] = None,
        index: t.Optional[t.List[str]] = None,
        measure: t.Optional[str] = None,
    ):
        self.qdrant_home = qdrant_home
        self.embedder = embedder
        self.dimensions = dimensions
        self.ks_backup_dir = ks_backup_dir
        self.vectors_config = VectorParams(
            size=CFG.embedder.size, distance=Distance.COSINE
        )

        self.measure = measure
        self._client: AsyncQdrantClient = AsyncQdrantClient(
            CFG.vector_db.qdrant.main_url, 
            port=CFG.vector_db.qdrant.port, 
            timeout=CFG.vector_db.qdrant.timeout,
        )
        self.timer = Timer()
        self.embed_lock = Lock()

    async def _create_collection(self, collection: str, recreate_collection: bool = False):
        collections_response: CollectionsResponse = await self._client.get_collections()
        qdrant_collections = collections_response.collections
        collection_exists = False
        if len(qdrant_collections) > 0:
            for qdrant_collection in qdrant_collections:
                if qdrant_collection and qdrant_collection.name == collection:
                    collection_exists = True
        if (not collection_exists) or recreate_collection:
            result = await self._client.recreate_collection(
                collection_name=collection,
                vectors_config=self.vectors_config,
            )
            return result
        return None

    def __len__(self) -> int:
        return -1;

    async def delete(
        self, 
        uuids: list[str],
        collection: str
    ) -> None:
        await self._client.delete(
            collection_name=collection, 
            points_selector=PointIdsList(points=uuids),
        )
     
    async def commit(
        self, 
        doc: KnapDocument, 
        collection: str, 
        uuids: list[str],
    ) -> UpsertStats:
        # TODO: time everything about this function. 
        # Gotta optimize this fn
        await self._create_collection(collection)
        self.timer.start('total')

        upsert_stats = UpsertStats()

        for chunk in doc:
            embed_cols = doc.embed_cols()
            metadata_cols = doc.metadata_cols()

            # print(f"chunk: {chunk}")
            if isinstance(chunk, pa.RecordBatch):
                chunk = chunk.to_pandas()
                df = chunk.where(pd.notnull(chunk), '')
                df = df.replace({np.nan: ''})
                logger.debug(f"df at start: {df}")
            else:
                df = pd.DataFrame.from_dict(chunk)

            if len(uuids) == 0:
                uuids = df['uuid'].tolist()
                df = df.rename(columns={'uuid': 'id'})
            else:
                df['id'] = uuids

            # print(f"processing {len(df)} rows...")

            self.timer.start('hashes')
            df['embed_hash'] = df.apply(util.hash_row, axis=1, cols=embed_cols, timer=self.timer)
            df['metadata_hash'] = df.apply(util.hash_row, axis=1, cols=metadata_cols, timer=self.timer)
            self.timer.end('hashes')

            # print(f"embed_cols: {embed_cols}")
            # print(f"metadata_cols: {metadata_cols}")
            logger.debug(f"df later: {df}")

            curr_points: list[Record] = await self._client.retrieve(
                collection_name=collection,
                ids=uuids,
                with_payload=True,
                with_vectors=True,
            )
            curr_points_df = self._build_curr_points_df(curr_points)
            logger.debug(f"curr_points_df: {curr_points_df}")

            self.timer.start('diff_embeddings')
            diff_embed_df = self._get_diff_rows_for_col(df, curr_points_df, 'embed_hash')
            logger.debug(f"diff_embed_rows: \n{diff_embed_df}")

            with self.embed_lock:
                diff_embeddings = create_embeddings(llm=self.embedder, content=self._content_to_embed(diff_embed_df, embed_cols))
                if diff_embeddings:
                    logger.debug(f"diff_embeddings: {len(diff_embeddings)} x {len(diff_embeddings[0])}")

            self.timer.end('diff_embeddings')

            if not curr_points_df.empty:
                # TODO: this can be further optimized. If the row appears in diff_embed_df, we don't need to 
                # upsert it here. It'll get upserted in the embed_batch.
                self.timer.start('build_metadata_batch')
                diff_metadata_df = self._get_diff_rows_for_col(df, curr_points_df, 'metadata_hash')
                diff_metadata_df = diff_metadata_df.set_index('id')
                curr_points_df = curr_points_df.set_index('id')
                curr_points_df = curr_points_df.reindex(diff_metadata_df.index)
                diff_metadata_df['vector'] = curr_points_df['vector']
                diff_metadata_df = diff_metadata_df[diff_metadata_df['vector'].notna()]
                diff_metadata_df = diff_metadata_df.reset_index()
                curr_points_df = curr_points_df.reset_index()
                logger.debug(f"diff_metadata_rows: \n{diff_metadata_df}")

                try: 
                    metadata_batch: Batch = self._build_points_for_upsert(diff_metadata_df, embed_cols, metadata_cols, embeddings=None)
                except pa.lib.ArrowInvalid as e:
                    logger.error(f"ArrowInvalid exception converting to Batch from diff_metadata_df: \n{diff_metadata_df}. Exception: \n{e}")
                    uuids = []
                    continue
                self.timer.end('build_metadata_batch')

                if len(metadata_batch.vectors) > 0:
                    self.timer.start('metadata_upsert')
                    try: 
                        await self._client.upsert(collection_name=collection, points=metadata_batch, wait=True)
                        upsert_stats.record_upserts(
                            num_metadata_upserted=len(metadata_batch.ids), 
                            metadata_uuids=metadata_batch.ids,
                        )
                    except ResponseHandlingException as e:
                        logger.error(f"Recorded exception upserting: \n{e}")
                        uuids = []
                        continue
                    except Exception as e:
                        logger.error(f"Recorded exception upserting. General exception: \n{e}")
                        uuids = []
                        continue
                    self.timer.end('metadata_upsert')

            if not diff_embed_df.empty:
                self.timer.start('build_embed_batch')
                try: 
                    embed_batch: Batch = self._build_points_for_upsert(diff_embed_df, embed_cols, metadata_cols, diff_embeddings)
                except pa.lib.ArrowInvalid as e:
                    logger.error(f"Exception converting to Batch from diff_embed_df: \n{diff_embed_df}. Exception: {e}")
                    uuids = []
                    continue
                self.timer.end('build_embed_batch')
                if len(embed_batch.vectors) > 0:
                    self.timer.start('embed_upsert')
                    try:
                        await self._client.upsert(collection_name=collection, points=embed_batch, wait=True)
                        upsert_stats.record_upserts(
                            num_embeddings_upserted=len(embed_batch.ids), 
                            embeddings_uuids=embed_batch.ids,
                        )
                    except ResponseHandlingException as e:
                        logger.error(f"Recorded exception upserting: {e}")
                        uuids = []
                        continue
                    except Exception as e:
                        uuids = []
                        logger.error(f"Recorded exception upserting. General exception: {e}")
                        continue
                    self.timer.end('embed_upsert')
            uuids = []
            logger.debug(f'timer: {self.timer}')
        self.timer.end('total')

        upsert_stats.record_upserts(time=self.timer.get('total'))
        return upsert_stats

    def _validate_embeddings(self, embeddings: list[list[float]]):
        if not isinstance(embeddings, list):
            raise KnapsackException(f"Expected embeddings to be list, got {type(embeddings)}. embeddings: {embeddings}")
        for embedding in embeddings:
            if not isinstance(embedding, list) and not isinstance(embedding, np.ndarray):
                raise KnapsackException(f"Expected embedding to be list or np.ndarray, got {type(embedding)}. embedding: {embedding}")
            # for num in embedding:
            #     if not isinstance(num, float):
            #         raise KnapsackException(f"Expected embedding num to be float, got {type(num)}. num: {num}")

    def _build_points_for_upsert(
        self, 
        df: pd.DataFrame, 
        embed_cols: list[str],
        metadata_cols: list[str],
        embeddings: list[list[float]] | None,
    ) -> Batch:
        if not 'id' in df.columns: 
            logger.error(f"'id' not a column in df: {df}.")
            raise ValueError(f"'id' not a column in df: {df}")
        ids = df['id'].tolist()
        if not isinstance(ids, list):
            raise ValueError("ids expected to be a list.")
        if embeddings is None:
            if not 'vector' in df.columns:
                logger.error(f"embeddings is None, but 'vector' not a column in df: \n{df}.")
                raise ValueError(f"embeddings is None, but 'vector' not a column in df: \n{df}.")
            embeddings = df['vector'].tolist()

        self._validate_embeddings(embeddings)

        # logger.debug(f"DF TO UPSERT: \n{df}")
        table = pa.Table.from_pandas(df)
        data = table.to_pydict()
        payloads = self._construct_payloads(data, metadata_cols)
        # logger.debug(f"PAYLOADS: \n{payloads}")
        
        logger.debug(f"ABOUT TO COMMIT final payloads: num: {len(payloads)}")
        logger.debug(f"ABOUT TO COMMIT: uuids: num: {len(ids)}, len embed: {len(embeddings)}")
        return Batch(ids=ids, payloads=payloads, vectors=embeddings)

    def _build_curr_points_df(self, curr_points: list[Record]) -> pd.DataFrame:
        # curr_points_df = pd.DataFrame.from_dict([{k: v for k, v in p['payload'].items()} for p in curr_points])
        if len(curr_points) == 0:
            return pd.DataFrame()
        flattened_data = []
        for p in curr_points:
            data_point = dict(p)
            flattened_data_point = {}
            if "payload" in data_point: 
                payload_data = data_point.pop('payload')
                flattened_data_point = {**data_point}
                if "ks" in payload_data:
                    flattened_data_point |= payload_data["ks"]  # merge ks metadata dictionary, if it exists
                    payload_data.pop("ks")
                flattened_data_point |= payload_data  # merge client payload dictionaries
                flattened_data.append(flattened_data_point)
        # print(f"flattened data: {flattened_data}")
        return pd.DataFrame(flattened_data)

    def _get_diff_rows_for_col(self, df1: pd.DataFrame, df2: pd.DataFrame, col: str) -> pd.DataFrame:
        # NOTE: the length and indices of df1 and df2 must match
        if df2.empty or col not in df2.columns:
            return df1

        df1 = df1.set_index('id')
        df2 = df2.set_index('id')
        df2 = df2.reindex(df1.index)

        # logger.debug(f"df1: \n{df1}")
        # logger.debug(f"df2: \n{df2}")

        result = df1[df1[col] != df2[col]]
        if not isinstance(result, pd.DataFrame): 
            logger.error("Error in _get_diff_rows_for_col.\ndf1: {df1}\n\ndf2: {df2}\n\ncol: {col}")
            raise ValueError("Expected DataFrame as a result in _get_diff_rows_for_col.")
        df1 = df1.reset_index()
        df2 = df2.reset_index()
        return result.reset_index().rename(columns={'index': 'id'})

    def _construct_payloads(self, data: pa.Table, metadata_cols: list[str]) -> list[dict]:
        payloads = []
        num_points = len(next(iter(data.values())))
        for i in range(num_points): 
            payload = {}
            ks_data = {}
            for k, v in data.items():
                if k == 'embed_hash': 
                    ks_data['embed_hash'] = v[i]
                elif k == 'metadata_hash':
                    ks_data['metadata_hash'] = v[i]
                elif (k != 'uuid' and k != 'vector' and k in metadata_cols) :
                    payload[k] = v[i]   
            payload['ks'] = ks_data
            payloads.append(payload)
        return payloads

    def _append_new_columns(self, collection: str, new_data: pd.DataFrame, metadata_cols: list[str]):
        pass

    async def find_by_uuid(
        self,
        uuids: list[str],
        collection: str,
        with_vector: bool = False,
        with_payload: bool = True,
    ) -> dict[dict[str, t.Any]]:
        """
        Returns a dict of dicts of the form:
        {
            uuid1: {'payload': payload, 'vector': vector},
            uuid2: ...,
            ...
        }
        """
        points: list[Record] = await self._client.retrieve(
            collection_name=collection,
            ids=uuids,
            with_payload=with_payload,
            with_vectors=with_vector,
        )
        results = {
            p.id: {
                'payload': p.payload,
                'vector': p.vector if with_vector else ""
            } for p in points
        }
        return results

    async def find_nearest_from_id(
        self,
        _id,
        collection: str,
        # return_columns: list[str] = ['uuid'],
        n: int = 100,
        filter: t.Optional[str] = None,
        return_original: bool = False,
        with_vector: bool = False,
    ) -> list[dict[str, t.Any]]:
        points = await self.find_by_uuid([_id], collection)
        point = points.get(_id, None)
        if point is None:
            return []
        return await self.find_nearest_from_array(
            h=point['vector'], 
            collection=collection, 
            # return_columns=return_columns, 
            return_original=return_original,
            filter=filter,
            n=n,
            with_vector=with_vector,
        )

    async def find_nearest_from_array(
        self,
        h: np.typing.ArrayLike,
        collection: str,
        n: int = 100,
        # return_columns: list[str] = ['uuid'],
        return_original: bool = False,
        with_vector: bool = False,
        filter: t.Optional[str] = None,
    ) -> list[dict[str, t.Any]]:
        self.timer.start('semantic_search')
        scored_points = await self._client.search(
            collection_name=collection,
            query_vector=h,
            query_filter=convert_sql_where_to_qdrant_filter(filter),
            limit=n if return_original else n+1,
            with_vectors=with_vector,
        )
        result = []
        for s in scored_points:
            if len(result) >= n: 
                break
            if ((s.score < 1) or return_original):
                result.append({
                    'vector': s.vector if with_vector else "",
                    'payload': s.payload,
                    'id': s.id,
                    'score': s.score,
                })
        result.sort(key=lambda d: d['score'], reverse=True)
        self.timer.end('semantic_search')
        return result

    async def list_vectors(
        self,
        collection: str, 
        num_results: int = 20,
        with_payload: bool = True,
        filter: t.Optional[str] = None,
        page: str | None = None,
    ):
        scroll_response = await self._client.scroll(
            collection_name=collection,
            scroll_filter=convert_sql_where_to_qdrant_filter(filter),
            limit=num_results,
            with_payload=with_payload,
            with_vectors=False,
            offset=page,
        )
        if isinstance(scroll_response, tuple):
            return {"vectors": [dict(v) for v in scroll_response[0]], "next_page": scroll_response[1]}
        else: 
            logger.debug(f"scroll_response wasn't tuple: {scroll_response}")
            return {}

    def _content_to_embed(
        self, 
        data: t.Union[pd.DataFrame, dict], 
        cols: list[str | None]
    ) -> list[str]:
        if len(cols) <= 0:
            return []
        if isinstance(data, pd.DataFrame):
            for col in cols:
                if 'embed_final' not in data:
                    data['embed_final'] = f"{col}: " + data[col].astype(str) + "\n\n"
                else:
                    data['embed_final'] += f"{col}: " + data[col].astype(str) + "\n\n"
            return data['embed_final'].tolist()
        else:
            list_length = len(next(iter(data.values())))

            results = []
            for i in range(list_length):
                concatenated = '\n\n'.join(data[col][i] for col in cols)
                results.append(concatenated)
            return results
    
    async def _create_snapshots(self, collection: str) -> list[str]:
        snapshot_urls = []
        for node_url in CFG.vector_db.qdrant.nodes:
            logger.debug(f"CREATE SNAPSHOT NODE URL : {node_url}")
            node_client = AsyncQdrantClient(node_url, timeout=CFG.vector_db.qdrant.timeout)
            try:
                snapshot_info = await node_client.create_snapshot(collection_name=collection)
                logger.debug(f"snapshot_info : {snapshot_info}")
            except Exception as e:
                logger.exception(f"Couldn't create snapshot for node: {node_url}")
                continue
            if snapshot_info:
                snapshot_url = f"{node_url}/collections/{collection}/snapshots/{snapshot_info.name}"
                snapshot_urls.append(snapshot_url)
        return snapshot_urls

    async def _download_snapshots(self, backup_name: str, snapshot_urls: list[str], collection: str):
        local_snapshot_paths = []
        for snapshot_url in snapshot_urls:
            snapshot_name = os.path.basename(snapshot_url)
            local_snapshot_dir = self.ks_backup_dir / Path(backup_name)
            local_snapshot_dir.mkdir(parents=True, exist_ok=True)
            local_snapshot_path = local_snapshot_dir / Path(snapshot_name)

            logger.debug(f"snapshot_url : {snapshot_url}")
            snapshot_location = self.qdrant_home / Path("snapshots") / Path(collection) / snapshot_name
            logger.debug(f"snapshot_location : {snapshot_location}")
            logger.debug(f"exists? : {snapshot_location.exists()}")
            shutil.move(snapshot_location, local_snapshot_path)
            # This is getting killed - assumedly due to an OOM error -
            # on a machine with 16GB RAM and a 5GB collection.
            # response = requests.get(snapshot_url, timeout=3600)

            # logger.debug(f"snapshot_url response : {response}")
            # with open(local_snapshot_path, "wb") as f:
            #     response.raise_for_status()
            #     f.write(response.content)
            # logger.debug(f"wrote to path: {local_snapshot_path}")
            logger.debug(f"moved download to path: {local_snapshot_path}")

            local_snapshot_paths.append(local_snapshot_path)
        return local_snapshot_paths

    async def backup(self, name: str, collection: str) -> bool:
        try:
            snapshot_urls = await self._create_snapshots(collection)
            local_snapshot_paths = await self._download_snapshots(
                backup_name=name, snapshot_urls=snapshot_urls, collection=collection
            )
        except Exception: 
            logger.exception("Issue creating backups.")
            return False
        return True

    async def restore(self, collection: str, ks_backup_path: Path):
        # for node_url, snapshot_path in zip(QDRANT_NODES, local_snapshot_paths):
        if not ks_backup_path.is_dir():
            raise ValueError(f"{ks_backup_path} is not a valid Knapsack backup.")
        snapshots = [file for file in ks_backup_path.glob('*.snapshot')] 

        def extract_timestamp(file: Path):
            try:
                # Assuming qdrant snapshot filename format is 
                # 'prefix-random_number-YYYY-MM-DD-HH-MM-SS.snapshot'
                timestamp_str = file.name.split('-')[-6:]
                timestamp_str = '-'.join(timestamp_str).split('.')[0]
                return datetime.strptime(timestamp_str, "%Y-%m-%d-%H-%M-%S")
            except Exception:
                logger.exception("couldn't extract timestamp.")
                return datetime.min

        sorted_snapshots = sorted(snapshots, key=extract_timestamp, reverse=True)
        latest_snapshot = Path(ks_backup_path) / sorted_snapshots[0]

        payload = {
            "location": f"file://{latest_snapshot}"
        }

        snapshot_name = os.path.basename(latest_snapshot)
        for node_url in CFG.vector_db.qdrant.nodes:
            if CFG.vector_db.api_key is not None:
                response = requests.put(
                    f"{node_url}/collections/{collection}/snapshots/recover",
                    headers={
                        "api-key": CFG.vector_db.api_key,
                    },
                    json=payload,
                    timeout=3600,
                )
            else: 
                response = requests.put(
                    f"{node_url}/collections/{collection}/snapshots/recover",
                    json=payload,
                    timeout=3600,
                )
        if response.ok:
            logger.debug(f"Snapshot recovery initiated successfully: {collection} --- {ks_backup_path} --- {latest_snapshot}.")
        else:
            logger.debug(f"Snapshot NOT initiated successfully: {response.status_code} " + 
                         f"--- {collection} --- {ks_backup_path} --- {latest_snapshot}.")

    async def list_snapshots(self, collection: str) -> list[str]:
        snapshots: list[SnapshotDescription] = await self._client.list_snapshots(collection)
        return [s.name for s in snapshots]

    async def delete_snapshot(self, collection: str, snapshot_name: str) -> None:
        await self._client.delete_snapshot(collection, snapshot_name, wait=False)

    async def list_collections(self) -> list[str]:
        collections_response: CollectionsResponse = await self._client.get_collections()
        qdrant_collections = collections_response.collections
        result = []
        for qdrant_collection in qdrant_collections:
            if qdrant_collection:
                result.append(qdrant_collection.name)
        return result

    async def copy_collection(self, existing_collection: str, new_collection: str):
        await self._client.create_collection(
            collection_name=new_collection,
            vectors_config=self.vectors_config,
            init_from=InitFrom(collection=existing_collection),
            timeout=3600,
        )

    async def delete_collection(self, collection: str) -> None:
        await self._client.delete_collection(collection, timeout=3600)

    async def count_collection(self, collection: str) -> int:
        count_result = await self._client.count(collection_name=collection, exact=True)
        return count_result.count

    async def collection_info(self, collection: str):
        return await self._client.get_collection(collection)
