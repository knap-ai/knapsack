import tempfile
import typing as t
from enum import Enum
from pathlib import Path
from threading import Lock

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.csv as pcsv
import re

from knapsack.base.error import KnapsackException
from knapsack.base.logger import logger


class DataType(Enum):
    URL = "url"
    UNSTRUCTURED = "unstructured"
    CSV = "csv"
    GOOGLE_LINK = "google_link"
    OTHER = "other"

KB = 1024
MB = KB * KB


class KnapDocument(object):
    def __init__(self, 
        data: t.Union[list[dict[str, t.Any]], str, Path], 
        tags: dict[str, t.Any] = {},
        extra_metadata_keys: list[str] = [],
    ) -> None:
        """
        data - list of dict[str, Any], or Path - `data` contains either data
            to upsert, or a valid uri that points to data to upsert. 
        tags - dict[str, Any] - metadata about the data. Add an "embed" 
            tag that maps to a list of column names (or dict keys) that should be embedded. 
            Add a "metadata" tag to list columns/dict keys that should be added as
            filterable metadata per row. Columns can be listed in both "embed"
            and "metadata", in which case they will both be embedded 
            as well as in vector payload for filter search.

            For csv or a list of dictionaries, tags could look like this:
                {
                    "embed": ["description", "name", "short_description", "founders"],
                    "metadata": ["name", "created", "type_of_exit", ...]
                    "source": "my_data.csv"
                }
            Any other key-value pairs in `tags` will be inserted as metadata for every row;
            every row will have identical metadata. This could be helpful for denoting 
            attributes like source (a particular csv file, a web page, etc.) that produced 
            many individual vectors in Knapsack.
         """
        self.data = data
        self.total_uuids = 0
        self._uri: t.Union[Path, None] = None
        self.is_dict = False
        if isinstance(data, str) or isinstance(data, Path):
            self._uri = Path(data)
        else: 
            self.is_dict = True
        self.tags = tags
        self.extra_metadata_keys = extra_metadata_keys
        self._in_memory: bool = False
        self.tmp_parquet_file = None
        self.lock = Lock()

    @property
    def content(self) -> t.Any:
        if self.data_type == DataType.CSV:
            return str(self.data)
        elif self.data_type == DataType.UNSTRUCTURED:
            if self._in_memory:
                return self.data
            return str(self.data)
        return pa.Table.from_pylist(self.data)
    
    @property
    def source(self) -> str:
        if self.data_type == DataType.CSV and self.uri:
            return str(Path(self.uri).name)
        raise ValueError("attr 'source' not implemented for this DataType.")

    @property
    def uri(self) -> t.Union[Path, None]:
        return self._uri

    @property
    def data_type(self) -> DataType:
        if self.uri and self._is_valid_google_link(self.uri):
            return DataType.GOOGLE_LINK
        elif self.uri and self._is_valid_url(self.uri):
            return DataType.URL
        elif self.uri and self._has_unstructured_loader_suffixes(self.uri):
            return DataType.UNSTRUCTURED
        elif self.uri and self._ends_with_csv(self.uri):
            if not self.uri.exists():
                raise KnapsackException(f"File {self.uri} not found.")
            self.convert_csv_to_parquet()
            return DataType.CSV
        else:
            if self.uri and not self.uri.exists():
                raise KnapsackException(f"File {self.uri} not found.")
            return DataType.OTHER

    def convert_csv_to_parquet(self):
        with self.lock:
            if self.tmp_parquet_file is None:
                logger.info(f'converting csv to parquet...: {self.uri}')
                table = pcsv.read_csv(self.uri)
                self.tmp_parquet_file = tempfile.NamedTemporaryFile(suffix='.parquet', delete=True)
                pq.write_table(table, self.tmp_parquet_file.name)

    def uuids(self) -> t.Generator[list[str], None, None]:
        """
        Iterator over all uuids. yields list[str]
        """
        if self.data_type == DataType.CSV:
            parquet_file = pq.ParquetFile(self.tmp_parquet_file.name)

            uuid_col = 'uuid'
            for batch in parquet_file.iter_batches(batch_size=16 * KB, columns=[uuid_col]):
                logger.debug(f"read new batch from .parquet")
                column = batch.column(0)
                uuids = [str(value.as_py()) for value in column]
                logger.info(f'grabbed {len(uuids)} from the .parquet')
                self.total_uuids += len(uuids)
                yield uuids

            # read_options = pcsv.ReadOptions(use_threads=True, block_size=8 * MB)
            # with pcsv.open_csv(self.uri, read_options=read_options) as reader:
            #     self.total_uuids = 0
            #     csv_uuid_col = 'uuid'
            #     for chunk in reader:
            #         logger.debug(f"read new chunk from .csv")
            #         if csv_uuid_col not in chunk.column_names:
            #             logger.debug(f"No 'uuid' column in .csv chunk: {chunk}")
            #             KnapsackException("No 'uuid' column in .csv")
            #         column = chunk[csv_uuid_col]
            #         uuids = [str(value) for value in column]
            #         # uuids += vals
            #         self.total_uuids += len(uuids)
            #         yield uuids
        # raise NotImplementedError("uuids() not implemented for this data type.")


    def embed_cols(self) -> list[str | None]:
        if isinstance(self.tags, dict) and 'embed' in self.tags:
            return self.tags['embed']
        return []

    def metadata_cols(self) -> list[str]:
        if isinstance(self.tags, dict):
            return self.tags.get('metadata', []) + list(self.extra_metadata_keys)
        return []

    def to_prompt(self) -> str:
        """
        Subclasses should overwrite this method
        to return relevant information for a prompt, 
        depending on what the data source is.

        See SlackDocument for an example.
        """

        # TODO: this will probably become important again
        # once LLM's are introduced.
        return ""

    def _has_unstructured_loader_suffixes(self, uri: t.Union[str, Path]):
        valid_suffixes = ['.pdf', '.txt', '.html', '.xml', '.ppt']
        pattern = r'\.(' + '|'.join(
            re.escape(suffix) for suffix in valid_suffixes
        ) + r')$'
        return re.search(pattern, str(uri), re.IGNORECASE) is not None

    def _ends_with_csv(self, uri: t.Union[str, Path]):
        pattern = r'\.csv$'
        return re.search(pattern, str(uri), re.IGNORECASE) is not None

    def _is_valid_url(self, uri: t.Union[str, Path]):
        pattern = r'^(https?|ftp)://[^\s/$.?#].[^\s]*$'
        return re.match(pattern, str(uri)) is not None

    def _is_valid_google_link(self, uri: t.Union[str, Path]):
        # logger.info(f'_is_valid_google_link: {uri}')
        # Regular expression pattern to match valid Google
        # Drive/Docs URLs with "Anyone can view" permission
        pattern = r'(https:\/\/docs\.google\.com\/' + \
            r'(file\/d\/|document\/d\/|drive\/folders\/)' + \
            r'[a-zA-Z0-9_-]+\/?[a-zA-Z0-9_-]*\?usp=sharing)'
        match = re.match(pattern, str(uri))

        if match:
            return True
        else:
            return False

    def __iter__(self) -> pa.Table:
        if self.is_dict:
            yield self.data
            return
        if self.uri:
            logger.info(f'Document uri: {self.uri}')
        if self.data_type == DataType.UNSTRUCTURED:
            display_len: int = max(len(self.content), 50)
            logger.info(f'Document content snippet: {self.content[display_len]}')
        if self.data_type == DataType.CSV:
            logger.info(f'Document content snippet: {self.content[50:]}')
        logger.info(f'Document type: {self.data_type}')
        i = 0

        if self.data_type == DataType.CSV:
            parquet_file = pq.ParquetFile(self.tmp_parquet_file.name)
            for batch in parquet_file.iter_batches(batch_size=256):
                yield batch

            # yield batch
            # read_options = pcsv.ReadOptions(use_threads=True, block_size=128 * KB)
            # parse_options = pcsv.ParseOptions(newlines_in_values=True)
            # with pcsv.open_csv(self.uri, read_options=read_options, parse_options=parse_options) as reader:
            #     for chunk in reader:
            #         # i += 1
            #         # if i == 1:
            #         #     continue
            #         yield chunk
        elif self.data_type == DataType.UNSTRUCTURED:
            pass
        else: 
            raise ValueError("Data type chunking error.")

    def __len__(self) -> int:
        return self.total_uuids
