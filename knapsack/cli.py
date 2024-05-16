import os
import typing as t
from pathlib import Path
from subprocess import Popen

import requests
import typer
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

import knapsack
from knapsack import Knapsack, logger, CFG
from knapsack.base.error import KnapsackException


typer_app = typer.Typer()
ks = Knapsack()
api = FastAPI()

origins = [
    "*"
]

api.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@typer_app.command()
def deploy(
    host: str = typer.Option(default="0.0.0.0"), 
    port: int = typer.Option(default=8888, help="The port on which to deploy knapsack.")
):
    global ks
    typer.echo("Deploying knapsack.")

    try: 
        # pid_path = Path(os.path.expanduser(CFG.knapsack_dir)) / Path("gunicorn_pid.txt")

        # num_workers = 4
        # app_module = 'knapsack.cli:api'
        # command = [
        #     "gunicorn", 
        #     "-w", str(num_workers), 
        #     "-k", "uvicorn.workers.UvicornWorker", 
        #     "-b", f"{host}:{port}",
        #     "--timeout", "16000",
        #     app_module,
        # ]
        # ks_process: Popen = Popen(command)
        # 
        # with open(str(pid_path), "w") as file:
        #     file.write(str(ks_process.pid))
        uvicorn.run(api, host=host, port=port, workers=1)

    except Exception as e:
        logger.exception(e)


@typer_app.command()
def shutdown(
    host: str = typer.Option(default="0.0.0.0"), 
    port: int = typer.Option(default=8888, help="The port on which to deploy knapsack.")
):
    # necessary work-around because of Uvicorn's lack of simple shutdown support.
    requests.get("http://" + host + ":" + str(port) + "/api/knapsack/shutdown")

@typer_app.command()
def run_connectors():
    global ks
    typer.echo("Running connectors...")
    ks.run()

@typer_app.command()
def search(
    query: str, collection: str, num_results: int = 20,
):
    global ks
    typer.echo("Searching Knapsack...")
    try:
        ks.semantic_search(
            query=query, 
            collection=collection,
            num_results=num_results,
            filter=None
        )
    except Exception as e:
        logger.error("Exception thrown in semantic_search: ", e)


def _handle_knapsack_exception(e: Exception) -> None:
    logger.exception(e)
    raise HTTPException(status_code=500, detail="Item not found")


def _val_collection(v):
    if not isinstance(v, str):
        raise ValueError("Param 'collection' must be a string.")


def _val_ids(ids: list[t.Union[str, int]]):
    for id in ids:
        # if isinstance(id, str):
            # if not is_valid_uuid(id):
            #     raise ValueError(f"Param 'id' is invalid. String ids can be urn, simple, or hyphenated. " + 
            #                      "Please see documentation for valid values.")
        if not isinstance(id, int):
            raise ValueError(f"Param 'ids' is invalid. Must either be a list of IDs, or an ID (string or int). " + 
                             "Please see documentation for details on valid values.")


class LearnRequest(BaseModel):
    data: t.Union[list[dict[str, t.Any]], str, Path]
    collection: str
    tags: dict[str, t.Any]
    ids: list[str] | list[int]
    wait: bool

    @classmethod
    def _val_data_dict(cls, d):
        for k, _ in d.items():
            if not isinstance(k, str):
                raise ValueError(f"Param 'data' is invalid. Key '{k}' is not a string. " + 
                                 "Please see documentation for valid values.")

    @field_validator('data')
    def data_val(cls, v):
        if isinstance(v, list):
            for elem in v:
                cls._val_data_dict(elem)
        elif isinstance(v, dict):
            raise ValueError("Param 'data' is invalid - it is neither list nor string nor Path. " + 
                             "Please see documentation for valid values.")
        elif not isinstance(v, str) and not isinstance(v, Path):
            raise ValueError("Param 'data' is invalid - it is neither list nor string nor Path. " + 
                             "Please see documentation for valid values.")
        return v

    @field_validator('collection')
    def collection_val(cls, v):
        _val_collection(v)
        return v

    @field_validator('tags')
    def tags_val(cls, v):
        if not isinstance(v, dict):
            raise ValueError(f"Param 'tags' must be a dict.")
        for k, _ in v.items():
            if not isinstance(k, str):
                raise ValueError(f"Param 'tags' is invalid. Key '{k}' is not a string. " + 
                                 "Please see documentation for valid values.")
        return v


class SemanticSearchRequest(BaseModel):
    query: t.Union[str, dict]
    collection: str
    num_results: t.Optional[int] = 20
    filter: str | None = None
    with_vector: t.Optional[bool] = False


class RetrieveRequest(BaseModel):
    uuids: list[str]
    collection: str
    with_vector: t.Optional[bool] = False

    # @field_validator('ids')
    # def ids_val(cls, v):
    #     if isinstance(v, list):
    #         _val_ids(v)
    #         return v
    #     else:
    #         raise ValueError(f"Param 'ids' is invalid. Must be a list of IDs (list of UUID string or 64-bit int). " + 
    #                          "Please see documentation for details on valid values.")

    @field_validator('collection')
    def collection_val(cls, v):
        _val_collection(v)
        return v


class VectorSearchRequest(BaseModel):
    vector: list[float]
    collection: str
    num_results: t.Optional[int] = 20
    return_original: t.Optional[bool] = False
    filter: str | None = None
    with_vector: t.Optional[bool] = False

    @field_validator('collection')
    def collection_val(cls, v):
        _val_collection(v)
        return v


class VectorCompareRequest(BaseModel):
    uuids1: list[str]
    collection1: str
    uuids2: list[str]
    collection2: str
    num_results: t.Optional[int] = 20
    return_original: t.Optional[bool] = False
    with_payload: t.Optional[bool] = True
    with_vector: t.Optional[bool] = False

    @field_validator('collection1')
    def collection_val(cls, v):
        _val_collection(v)
        return v

    @field_validator('collection2')
    def collection2_val(cls, v):
        _val_collection(v)
        return v


class DeleteRequest(BaseModel):
    uuids: list[str]
    collection: str

    # @field_validator('ids')
    # def ids_val(cls, v):
    #     if isinstance(v, list):
    #         _val_ids(v)
    #         return v
    #     else:
    #         raise ValueError(f"Param 'ids' is invalid. Must be a list of IDs (list of UUID string or 64-bit int). " + 
    #                          "Please see documentation for details on valid values.")

    @field_validator('collection')
    def collection_val(cls, v):
        _val_collection(v)
        return v


class FindByRequest(BaseModel):
    uuids: list[str]
    collection: str

    @field_validator('collection')
    def collection_val(cls, v):
        _val_collection(v)
        return v


class VectorListRequest(BaseModel):
    collection: str
    num_results: t.Optional[int] = 200
    with_payload: t.Optional[bool] = True
    filter: str | None = None
    page: t.Optional[str] = None

    @field_validator('collection')
    def collection_val(cls, v):
        _val_collection(v)
        return v


class BackupRequest(BaseModel):
    name: str
    collection: str

    @field_validator('collection')
    def collection_val(cls, v):
        _val_collection(v)
        return v


class BackupAllRequest(BaseModel):
    name: str


class RestoreRequest(BaseModel):
    snapshot_path: str
    collection: str

    @field_validator('snapshot_path')
    def ks_backup_path_val(cls, v):
        if not Path(v).exists():
            raise ValueError(f"Param 'snapshot_path' is invalid. It must be a valid path to a snapshot.")
        return v

    @field_validator('collection')
    def collection_val(cls, v):
        _val_collection(v)
        return v


class EmbedRequest(BaseModel):
    content: list[str]


class CopyCollectionRequest(BaseModel):
    existing_collection: str
    new_collection: str


class DeleteCollectionRequest(BaseModel):
    collection: str


class CollectionInfoRequest(BaseModel):
    collection: str


class CountCollectionRequest(BaseModel):
    collection: str


class GetMessagesRequest(BaseModel):
    # user: 
    pass


class SendMessageRequest(BaseModel):
    msg_data: dict[str, t.Any]


class LlmRequest(BaseModel):
    prompt: str
    stream: bool = False


class ExtractTextRequest(BaseModel):
    source: str | Path 


@api.post("/api/knapsack/learn")
async def learn(req: LearnRequest):
    """
    Example:
    {
        data=csv_path, 
        collection="cb", 
        tags={
            embed: ["description"],
            metadata: ["name", "id", "type", "rank", "created_at"],
            ...
        }
    }
    """
    global ks
    response = {'message': 'failure'}
    try:
        result = ks.learn(
            data=req.data, collection=req.collection, tags=req.tags, ids=req.ids, wait=req.wait
        )
        response = {"message": "success", "upsert_stats": str(result)}
    except KnapsackException as e:
        logger.info(f"Sending KnapsackException to client: {e}")
        response['error'] = str(e)
    except Exception as e:
        response['error'] = str(e)
        logger.exception(e)
    return response


@api.post("/api/knapsack/semantic_search")
async def semantic_search(req: SemanticSearchRequest):
    """
    Example: 
    {
        query="biotech protein synthesis", 
        collection="cb_test",
        return_columns=['id', 'name', 'rank'],
        num_results=200,
        filter="rank <= 50000 and rank > 0",
    }
    """
    global ks
    response = {'message': 'failure'}
    try:
        num_results = 20 if req.num_results == None else req.num_results
        if isinstance(req.query, dict):
            formatted_pairs = [f"{key}:{value}" for key, value in req.query.items()]
            query = '\n'.join(formatted_pairs)
        else:
            query: str = str(req.query)

        results = ks.semantic_search(
            query=query, 
            collection=str(req.collection),
            num_results=num_results,
            filter=req.filter,
        )
        response = {"message": "success", "data": results}
    except KnapsackException as e:
        response['reason'] = str(e)
        logger.exception(e)
    except Exception as e:
        response['reason'] = str(e)
        logger.exception(e)
    return response


@api.post("/api/knapsack/retrieve")
def retrieve(req: RetrieveRequest):
    global ks
    response = {'message': 'failure'}
    try:
        with_vector = bool(req.with_vector)
        result = ks.retrieve(req.uuids, req.collection, with_vector)
        response = {'message': 'success', "data": result}
    except KnapsackException as e:
        response['reason'] = str(e)
        logger.exception(e)
    except Exception as e:
        response['reason'] = str(e)
        logger.exception(e)
    return response 


@api.post("/api/knapsack/vector_search")
def vector_search(req: VectorSearchRequest):
    global ks
    response = {'message': 'failure'}
    try:
        num_results = 20 if req.num_results == None else req.num_results
        return_original = bool(req.return_original)
        with_vector = bool(req.with_vector)
        results = ks.vector_search(
            req.vector, 
            req.collection, 
            num_results=num_results,
            return_original=return_original,
            filter=req.filter,
            with_vector=with_vector,
        )
        response = {"message": "success", "data": results}
    except KnapsackException as e:
        response['reason'] = str(e)
        logger.exception(e)
    except Exception as e:
        response['reason'] = str(e)
        logger.exception(e)
    return response


@api.post("/api/knapsack/vector_compare")
def vector_compare(req: VectorCompareRequest):
    global ks
    response = {'message': 'failure'}
    try:
        num_results = 20 if req.num_results == None else req.num_results
        return_original = bool(req.return_original)
        with_payload = bool(req.with_payload)
        with_vector = bool(req.with_vector) 
        results = ks.vector_compare(
            req.uuids1, 
            req.collection1, 
            req.uuids2, 
            req.collection2, 
            num_results=num_results,
            return_original=return_original,
            with_payload=with_payload,
            with_vector=with_vector,
        )
        response = {'message': 'success', 'data': results}
    except KnapsackException as e:
        response['reason'] = str(e)
        logger.exception(e)
    except Exception as e:
        response['reason'] = str(e)
        logger.exception(e)
    return response


@api.post("/api/knapsack/find_by")
async def find_by(req: FindByRequest):
    global ks
    response = {'message': 'failure'}
    try:
        if isinstance(req.uuids, list) and len(req.uuids) >= 0:
            record = ks.find_by(req.uuids, req.collection)
            response['data'] = record
            response['message'] = 'success' 
    except KnapsackException as e:
        logger.exception(e)
        response['reason'] = str(e)
    except Exception as e:
        logger.exception(e)
    return response


@api.post("/api/knapsack/vector_list")
async def vector_list(req: VectorListRequest):
    global ks
    response = {'message': 'failure'}
    num_results = 200 if req.num_results == None else req.num_results
    with_payload = bool(req.with_payload)
    try:
        vectors = ks.vector_list(
            collection=req.collection, 
            num_results=num_results,
            with_payload=with_payload,
            filter=req.filter,
            page=req.page
        )
        response['data'] = vectors
        response['message'] = 'success' 
    except KnapsackException as e:
        response['reason'] = str(e)
        logger.exception(e)
    except Exception as e:
        response['reason'] = str(e)
        logger.exception(e)
    return response


@api.post("/api/knapsack/delete")
async def delete(req: DeleteRequest):
    global ks
    response = {'message': 'failure'}
    try:
        ks.delete(req.uuids, req.collection)
        response = {'message': 'success'}
    except KnapsackException as e:
        response['reason'] = str(e)
        logger.exception(e)
    except Exception as e:
        response['reason'] = str(e)
        logger.exception(e)
    return response


@api.post("/api/knapsack/backup")
async def backup(req: BackupRequest):
    global ks
    response = {'message': 'failure'}
    try:
        backup: Path | None = ks.backup(req.name, req.collection)
        response = {"message": "success", "data": backup}
    except KnapsackException as e:
        response['reason'] = str(e)
        logger.exception(e)
    except Exception as e:
        response['reason'] = str(e)
        logger.exception(e)
    return response


@api.post("/api/knapsack/backup_all")
async def backup_all(req: BackupAllRequest):
    global ks
    response = {'message': 'failure'}
    try:
        collections = ks.list_collections()
        results = []
        for collection in collections:
            backup = ks.backup(req.name, collection)
            if backup: 
                results.append(backup)
        response = {"message": "success", "data": results}
    except KnapsackException as e:
        response['reason'] = str(e)
        logger.exception(e)
    except Exception as e:
        response['reason'] = str(e)
        logger.exception(e)
    return response


@api.post("/api/knapsack/restore")
async def restore(req: RestoreRequest):
    global ks
    response = {'message': 'failure'}
    try:
        ks.restore(req.collection, Path(req.snapshot_path))
        response = {"message": "success"}
    except KnapsackException as e:
        logger.exception(e)
        response['reason'] = str(e)
    except Exception as e:
        logger.exception(e)
    return response


@api.post("/api/knapsack/embed")
async def embed(req: EmbedRequest):
    response = {'message': 'failure'}
    try:
        vectors = ks.embed(req.content)
        response = {"message": "success", "vectors": vectors}
    except KnapsackException as e:
        logger.exception(e)
        response['reason'] = str(e)
    except Exception as e:
        logger.exception(e)
    return response


@api.post("/api/knapsack/usage_report")
async def usage_report():
    response = {'message': 'failure', 'reason': 'not currently supported.'}
    # try:
    #     resource_tracker = ResourceTracker()
    #     resource_tracker.record()
    #     response = {"message": "success", "data": resource_tracker.__dict__()}
    # except Exception as e:
    #     logger.exception(e)
    return response


@api.post("/api/knapsack/collection_info")
async def collection_info(req: CollectionInfoRequest):
    global ks
    response = {'message': 'failure'}
    try:
        collection_info = ks.collection_info(req.collection)
        response = {"message": "success", "data": collection_info}
    except KnapsackException as e:
        response['reason'] = str(e)
        logger.exception(e)
    except Exception as e:
        response['reason'] = str(e)
        logger.exception(e)
    return response


@api.post("/api/knapsack/list_collections")
async def list_collections():
    global ks
    response = {'message': 'failure'}
    try:
        collections = ks.list_collections()
        response = {"message": "success", "data": collections}
    except KnapsackException as e:
        response['reason'] = str(e)
        logger.exception(e)
    except Exception as e:
        response['reason'] = str(e)
        logger.exception(e)
    return response


@api.post("/api/knapsack/copy_collection")
async def copy_collection(req: CopyCollectionRequest):
    global ks
    response = {'message': 'failure'}
    try:
        ks.copy_collection(req.existing_collection, req.new_collection)
        response = {"message": "success"}
    except KnapsackException as e:
        response['reason'] = str(e)
        logger.exception(e)
    except Exception as e:
        response['reason'] = str(e)
        logger.exception(e)
    return response


@api.post("/api/knapsack/delete_collection")
async def delete_collection(req: DeleteCollectionRequest):
    global ks
    response = {'message': 'failure'}
    try:
        ks.delete_collection(req.collection)
        response = {"message": "success"}
    except KnapsackException as e:
        response['reason'] = str(e)
        logger.exception(e)
    except Exception as e:
        response['reason'] = str(e)
        logger.exception(e)
    return response


@api.post("/api/knapsack/count_collection")
async def count_collection(req: CountCollectionRequest):
    global ks
    response = {'message': 'failure'}
    try:
        count = ks.count_collection(req.collection)
        response = {"message": "success", "data": count}
    except KnapsackException as e:
        response['reason'] = str(e)
        logger.exception(e)
    except Exception as e:
        response['reason'] = str(e)
        logger.exception(e)
    return response

if __name__ == "__main__":
    typer_app()
