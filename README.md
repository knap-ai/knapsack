<img src="assets/knapsack-logo.svg" width="100" height="100">

Knapsack 🎒 - Data connectors for fast, private AI.
---

## Title and Description 📝
Knapsack 🎒 is a open-source service that hosts and runs fast, private connectors for data to AI projects. Much like Glean or Perplexity, Knapsack 🎒 powers intelligent search and next-gen AI applications, but with an emphasis on community, privacy, and security.  

## Installation and Setup ⚙️
Knapsack connectors fetch data, transform, and load that data into a VectorDB backend. Efficient, secure, and easy data handling is our bread and butter. To this end, Knapsack 🎒 provides a simple, easy-to-use API for data connectors and the service can be launched via Docker.

To get started with Knapsack 🎒, ensure you have Docker installed on your machine. You can launch the service using Docker Compose:

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/your-repo/knapsack.git
   cd knapsack

2. Run it as a FastAPI server:
```bash
python -m knapsack.cli deploy --port 8888
```

3. Or utilize it directly as a library:
```python
from knapsack import Knapsack
ks = Knapsack()
ks.run()
```

## Roadmap 🔨

- [x] ArXiv, Base connector
- [x] Qdrant integration
- [x] Caching of certain APIs
- [x] Smart upsert to vector DB (hashed values, only upsert on change)
- [x] Scheduling
- [ ] GSuite
- [ ] BioArXiv
- [ ] PubMed

### VectorDB Integrations
- [x] Qdrant
- [ ] Milvus
- [ ] Weaviate
- [ ] Chroma


## How to Contribute 🤝
We welcome contributions from the community! Currently, we are particularly interested in adding more connectors. If you have developed a connector that could be useful to others, please consider submitting a pull request.

For those interested in public data, Knapsack 🎒 hosts publicly-accessible datasets, such as data derived from ArXiv, available for search and GPT chat via the Knapsack Desktop application. If you want to contribute to Knapsack 🎒 could, please reach out via our GitHub issues or file a pull request. Knap will host any new connectors that connect public data so that all users can take benefit from the abilities of LLM chat and search.

## License Information 📄
Knapsack 🎒 is released under the GNU General Public License v3.0. For more information, please refer to the LICENSE file in the repository.

Feel free to explore, modify, and distribute any part of Knapsack's 🎒 codebase. If you use Knapsack 🎒 in your research or projects, please consider citing it.
