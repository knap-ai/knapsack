from Bio import Entrez
from datetime import datetime, timedelta
from tqdm import tqdm


class PubMedConnector(BaseConnector):
    def __init__(self, email, **kwargs):
        super().__init__(**kwargs)
        self.email = email
        Entrez.email = self.email  # Set the email for Entrez API usage

    def fetch(self):
        for start_date, end_date in self.calculate_week_ranges():
            article_ids = self.search_articles(start_date, end_date)
            articles = self.fetch_details(article_ids)
            with tqdm(total=len(articles['PubmedArticle']), desc=f"Fetching from {start_date} to {end_date}") as pbar:
                for article in articles['PubmedArticle']:
                    title = article['MedlineCitation']['Article']['ArticleTitle']
                    abstract = article['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', ['No abstract available'])[0]
                    pbar.set_description(f"Processing: {title[:30]}...")
                    pbar.update(1)
                    self.store(title, abstract)
            time.sleep(1)  # Sleep to simulate processing delay

    def search_articles(self, start_date, end_date):
        search_term = f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
        handle = Entrez.esearch(db="pubmed", term=search_term, retmax=1000)
        record = Entrez.read(handle)
        handle.close()
        return record["IdList"]

    def fetch_details(self, id_list):
        ids = ','.join(id_list)
        handle = Entrez.efetch(db="pubmed", id=ids, retmode="xml")
        records = Entrez.read(handle)
        handle.close()
        return records

    def store(self, title, abstract):
        data = f"Title: {title}\nAbstract: {abstract}"
        # Simulate embedding in Qdrant database
        print(f"Storing data: {data}")


# def search_articles(start_date, end_date):
#     Entrez.email = "info@knap.ai"  # Always provide your email
#     search_term = f'("{start_date}"[Date - Publication] : "{end_date}"[Date - Publication])'
#     
#     handle = Entrez.esearch(
#         db="pubmed", term=search_term, retmax=1000
#     )  # Adjust retmax as needed
#     record = Entrez.read(handle)
#     handle.close()
#     
#     return record["IdList"]
# 
# def fetch_details(id_list):
#     ids = ','.join(id_list)
#     Entrez.email = "info@knap.ai"  # Always provide your email
#     handle = Entrez.efetch(
#         db="pubmed", id=ids, retmode="xml"
#     )
#     records = Entrez.read(handle)
#     handle.close()
#     
#     return records
# 
# # Calculate yesterday's date
# yesterday = (datetime.now() - timedelta(1)).strftime("%Y/%m/%d")
# article_ids = search_articles(yesterday, end_date=datetime.now())
# 
# # Fetch details of the articles
# articles = fetch_details(article_ids)
# 
# # Example of how to process the articles
# # This part can be customized based on the needed metadata
# for article in articles['PubmedArticle']:
#     print(f"Title: {article['MedlineCitation']['Article']['ArticleTitle']}")
#     # Add more fields as needed
