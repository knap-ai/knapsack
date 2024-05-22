from typing import Any


class BaseConnector:
    EMBED_TAG_PROPERTY_NAME = 'embed'
    METADATA_TAG_PROPERTY_NAME = 'metadata'

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
    
    def get_extra_metadata(self, tags: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in tags.items() if key != 'metadata' and key != 'embed'}

    # def store(self, title, abstract):
    #     data = f"Title: {title}\nAbstract: {abstract}"
    #     # Simulate embedding in Qdrant database
    #     raise NotImplementedError("")
