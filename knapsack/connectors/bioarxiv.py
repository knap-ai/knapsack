import requests
from datetime import datetime, timedelta

class BioArXivConnector(BaseConnector):
    def fetch(self):
        for start_date, end_date in self.calculate_week_ranges():
            response = self.make_request(start_date, end_date)
            if response and 'collection' in response:
                with tqdm(total=len(response['collection']), desc=f"Fetching from {start_date} to {end_date}") as pbar:
                    for article in response['collection']:
                        pbar.set_description(f"Processing: {article['title'][:30]}...")
                        pbar.update(1)
                        self.store(article['title'], article.get('abstract', 'No abstract available'))
                time.sleep(1)  # Sleep to simulate processing delay and not hammering the server

    def make_request(self, start_date, end_date):
        cursor = 0
        base_url = f"https://api.biorxiv.org/details/biorxiv/{start_date}/{end_date}/{cursor}"
        response = requests.get(base_url)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to fetch data: {response.status_code}")
            return None

    def formulate_query(self, start_date, end_date):
        base_url = f"https://api.biorxiv.org/details/biorxiv/{start_date}/{end_date}/0"
        return base_url


# def search_biorxiv(start_date, end_date, cursor=0, max_results=100):
#     base_url = f"https://api.biorxiv.org/details/biorxiv/{start_date}/{end_date}/{cursor}"
#     
#     response = requests.get(base_url)
#     if response.status_code == 200:
#         return response.json()
#     else:
#         print(f"Failed to fetch data: {response.status_code}")
#         return None
# 
# # Example usage:
# if __name__ == "__main__":
#     # Calculate start and end dates for the query
#     start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
#     end_date = datetime.now().strftime('%Y-%m-%d')
#     
#     result = search_biorxiv(start_date, end_date)
#     
#     if result:
#         for article in result['collection']:
#             print(f"Title: {article['title']}")
#             print(f"DOI: {article['doi']}")
#             print(f"Abstract: {article['abstract']}\n")
