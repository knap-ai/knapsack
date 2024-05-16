from typing import Any


class BaseConnector:
    def fetch(self):
        raise NotImplementedError("Each connector must implement its own fetch method.")

    def knapsack_tags(self) -> dict[str, Any]:
        """
        Return the tags associated with data from this source.
        Used for ex: querying the VectorDB for results from this
        connector.
        """
        raise NotImplementedError("Each connector must implement its own " +
                                  "knapsack_tags method.")
    # def store(self, title, abstract):
    #     data = f"Title: {title}\nAbstract: {abstract}"
    #     # Simulate embedding in Qdrant database
    #     raise NotImplementedError("")
