import typing as t

from knapsack import logger 


class UpsertStats(object):
    def __init__(self):
        self.stats: dict[str, t.Any] = {}
        self.stats['total_embeddings_upserted'] = 0
        self.stats['total_metadata_upserted'] = 0
        self.stats['embeddings_uuids'] = []
        self.stats['metadata_uuids'] = []
        self.stats['time'] = 0

    def record_upserts(
        self, 
        num_embeddings_upserted: int = 0, 
        num_metadata_upserted: int = 0, 
        embeddings_uuids: list[str] = [],
        metadata_uuids: list[str] = [],
        time: int = 0, 
    ) -> None:
        self.stats['total_embeddings_upserted'] += num_embeddings_upserted
        self.stats['total_metadata_upserted'] += num_metadata_upserted
        # self.stats['embeddings_uuids'] += embeddings_uuids
        # self.stats['metadata_uuids'] += metadata_uuids
        self.stats['time'] += time

    def save_to_disk(self):
        logger.info(f"UpsertStats: {self.stats}")

    def __str__(self) -> str:
        return str(self.stats)
