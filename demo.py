import argparse
from knapsack import Knapsack

import pprint as pp


def get_cli_args():
    parser = argparse.ArgumentParser(description='Knapsack')
    parser.add_argument('--query', type=str, help='Query to search for')
    return parser.parse_args()


knapsack = Knapsack()
# knapsack.run()

args = get_cli_args()
results = knapsack.semantic_search(
    query=args.query,
    collection="knapsack",
    num_results=20,
    filter="",
)

printer = pp.PrettyPrinter(indent=2)
for result in results:
     printer.pprint(results)
