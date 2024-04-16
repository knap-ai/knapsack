class BaseConnector:
    def __init__(self, start_date, end_date, max_results):
        self.start_date = start_date
        self.end_date = end_date if end_date != "now" else datetime.now().strftime('%Y-%m-%d')
        self.max_results = max_results
    
    def calculate_week_ranges(self):
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        delta = timedelta(days=7)
        while start < end:
            yield start.strftime('%Y-%m-%d'), (start + delta).strftime('%Y-%m-%d')
            start += delta
    
    def fetch(self):
        raise NotImplementedError("Each connector must implement its own fetch method.")
    
    def store(self, title, abstract):
        data = f"Title: {title}\nAbstract: {abstract}"
        # Simulate embedding in Qdrant database
        raise NotImplementedError("")
