from os import path, popen
from pathlib import Path
from typing import Any, List
from uuid import NAMESPACE_DNS, uuid5

from tqdm import tqdm

from knapsack.base.error import KnapsackException
from knapsack.connectors.base_connector import BaseConnector


class Entry(object):
    """This class represents one Git file entry"""

    def __init__(
        self, file_path: str, file_name: str, file_content: str, file_summary: str, file_creation_timestamp, file_modification_timestamp, file_size_in_mb
    ):
        self.file_path = file_path
        self.file_name = file_name
        self.file_content = file_content
        self.file_summary=file_summary
        self.file_creation_timestamp = file_creation_timestamp
        self.file_modification_timestamp = file_modification_timestamp
        self.file_size_in_mb = file_size_in_mb

    def to_dict(self) -> dict:
        return {
            "file_path": self.file_path,
            "file_name": self.file_name,
            "file_content": self.file_content,
            "file_summary":self.file_summary,
            "file_creation_timestamp": self.file_creation_timestamp,
            "file_modification_timestamp": self.file_modification_timestamp,
            "file_size_in_mb": self.file_size_in_mb,
        }

    def uuid(self) -> str:
        return str(uuid5(NAMESPACE_DNS, self.file_path))


def divide_chunks(list: List[Any], size: int):
    """
    Divide a list into chunks of specified size.

    Params:
        list: The list to be divided into chunks.
        size: The size of each chunk.

    Yields:
        List[Any]: A chunk of the original list of size 'size'.

    """
    for index in range(0, len(list), size):
        yield list[index : index + size]


class GitConnector(BaseConnector):
    def __init__(self, repository_path: str, project_name: str, **kwargs):
        self.repository_path_str = repository_path
        self.project_name = project_name
        super().__init__(**kwargs)

    def _get_versioned_files(self) -> List[str]:
        """
        Get current files versioned by git in the repository dir

        Returns:
            The list of versioned files' relative paths

        """
        has_git_instaled = "version" in popen("git --version").read()
        if not has_git_instaled:
            raise KnapsackException(
                "You need to have git installed in order to use the git connector."
            )
        files = (
            popen(
                f"git -C {self.repository_path_str} ls-tree --full-tree -r --name-only HEAD"
            )
            .read()
            .split("\n")
        )
        return [file for file in files if file.strip()]

    def fetch(self):
        repository_path = Path(self.repository_path_str)
        files = self._get_versioned_files()
        with tqdm(total=len(files)) as pbar:
            for batch in divide_chunks(files, 100):
                results = []
                for file in batch:
                    entry = self.convert_to_entry(
                        Path(path.join(repository_path, file))
                    )
                    pbar.update(1)
                    if entry:
                        results.append(entry)
                yield results

    def knapsack_tags(self) -> dict[str, Any]:
        """
        Return the tags associated with data from this source.
        Used for ex: querying the VectorDB for results from this
        connector.
        """
        return {
            "connector": "git",
            "repository_path": self.repository_path_str,
            "embed": ["file_name", "file_path", "file_content"],
            "metadata": ["file_content", "file_name", "file_path", "file_summary", "file_creation_timestamp", "file_modification_timestamp", "file_size_in_mb"],
        }

    def convert_to_entry(self, file: Path) -> Entry | None:
        try:
            file_content = file.read_text()
        except (UnicodeDecodeError, IsADirectoryError):
            return None
        file_path = file.absolute().as_posix()
        modification_timestamp = path.getmtime(file)
        creation_timestamp = path.getctime(file)
        file_size_in_mb = path.getsize(file) / 1024 / 1024

        return Entry(
            file_path=file_path,
            file_name=file.name,
            file_content=file_content,
            file_summary=file_content[:1000],
            file_creation_timestamp=creation_timestamp,
            file_modification_timestamp=modification_timestamp,
            file_size_in_mb=file_size_in_mb,
        )
