import typing as t
from datetime import datetime, timedelta


class BaseConnector:
    def __init__(
        self, start_date, end_date, interval: int,
    ):
        self.start_date = start_date
        self.end_date = end_date if end_date != "now" else datetime.now().strftime('%Y-%m-%d')
    
    def calculate_week_ranges(self):
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        delta = timedelta(days=1)
        while start < end:
            yield start.strftime('%Y%m%d')
            start += delta

    def calculate_day_ranges(self):
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        delta = timedelta(days=1)
        while start < end:
            next_day = start + delta
            yield start.strftime('%Y%m%d'), next_day.strftime('%Y%m%d')
            start = next_day

    def fetch(self):
        raise NotImplementedError("Each connector must implement its own fetch method.")
    
    def knapsack_tags(self) -> dict[str, t.Any]:
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
