from knapsack import CFG
import schedule
import time
from tqdm import tqdm

class Knapsack:
    def __init__(self):
        self.connectors = [self.initialize_connector(connector) for connector in CFG.connectors]

    def initialize_connector(self, connector_config):
        connector_name = connector_config['name']
        if connector_name == "ArXivConnector":
            return ArXivConnector(**connector_config['args'])
        elif connector_name == "BioArXivConnector":
            return BioArXivConnector(**connector_config['args'])
        else:
            raise ValueError(f"Unsupported connector: {connector_name}")

    def run(self):
        for connector in tqdm(self.connectors, desc="Running connectors"):
            schedule.every(connector.interval).minutes.do(connector.fetch)
        
        while True:
            schedule.run_pending()
            time.sleep(1)
