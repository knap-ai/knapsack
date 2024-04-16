import feedparser
from datetime import datetime, timedelta
from knapsack.connectors.base_connector import BaseConnector


class ArXivConnector(BaseConnector):
    def __init__(self, query, **kwargs):
        super().__init__(**kwargs)
        self.query = query
    
def __init__(self, query, **kwargs):
        super().__init__(**kwargs)
        self.query = query

    def fetch(self):
        for start_date, end_date in self.calculate_week_ranges():
            query_url = self.formulate_query(start_date, end_date)
            feed = feedparser.parse(query_url)
            with tqdm(total=len(feed.entries), desc=f"Fetching from {start_date} to {end_date}") as pbar:
                for entry in feed.entries:
                    pbar.set_description(f"Processing: {entry.title[:30]}...")
                    pbar.update(1)
                    self.store(entry.title, entry.get('summary', 'No abstract available'))
            time.sleep(1)  # Sleep to simulate processing delay and not hammering the server

    def formulate_query(self, start_date, end_date):
        base_url = "http://export.arxiv.org/api/query?"
        search_query = f'search_query={self.query}'
        start = f'start=0'
        max_results = f'max_results={self.max_results}'
        time_range = f'submittedDate:[{start_date}T00:00:00Z TO {end_date}T23:59:59Z]'
        query_url = f'{base_url}{search_query} AND {time_range}&{start}&{max_results}'
        return query_url

# def search_arxiv(start_date, end_date, query="all:electron", max_results=10):
#     # Format the query URL
#     base_url = "http://export.arxiv.org/api/query?"
#     search_query = f'search_query={query}'
#     start = f'start=0'
#     max_results = f'max_results={max_results}'
#     time_range = f'submittedDate:[{start_date}T00:00:00Z TO {end_date}T23:59:59Z]'
#     query_url = f'{base_url}{search_query} AND {time_range}&{start}&{max_results}'
#     
#     feed = feedparser.parse(query_url)
#     return feed
# 
# # Calculate start and end dates for the query
# start_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
# end_date = datetime.now().strftime('%Y%m%d')
# 
# # Fetch articles submitted in the last day
# # Adjust the query and max_results as needed
# feed = search_arxiv(start_date, end_date, max_results=5)
# 
# # Print article titles and URLs
# for entry in feed.entries:
#     print(f"Title: {entry.title}")
#     print(f"URL: {entry.link}\n")
