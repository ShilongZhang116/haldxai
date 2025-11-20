# ⭐ **HALDxAI – Human Aging and Longevity Database with AI**

*A scalable AI-driven platform for aging-related knowledge extraction, integration, and discovery*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Documentation](https://img.shields.io/badge/Documentation-latest-brightgreen)](docs/)

---

HALDxAI is an open-source platform designed for large-scale construction and analysis of aging-related knowledge graphs.
It integrates **445,000+ PubMed articles** and **19 biological databases**, combining natural language processing, machine learning, ontology alignment, and graph analytics to support research on aging biology, longevity mechanisms, and related biomedical discoveries.

---

# 🌟 Key Features

* **Entity Recognition & Normalization**
  LLM-based and ML-based NER pipelines for extracting aging-related biological entities.

* **Relation Extraction**
  Automated extraction of mechanistic, regulatory, genetic and phenotypic relationships from literature.

* **Knowledge Graph Construction**
  Multi-source integration across databases (HGNC, UniProt, CTD, PrimeKG, miRBase and others) into a unified Aging-KG.

* **Scoring & Ranking System**
  Multi-dimensional evaluation of entity importance, relationship strength and biological relevance.

* **Interactive Analysis & Visualization**
  Tools for graph exploration, neighborhood analysis, pathway tracing and network visualization.

* **Database Support**
  PostgreSQL for structured storage and Neo4j for graph querying and reasoning.

---

# 🚀 Getting Started

## Requirements

* Python 3.8 or later
* PostgreSQL 12+
* Neo4j 4.0+ (optional but recommended)
* Sufficient compute resources for LLM inference

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ShilongZhang116/haldxai.git
cd haldxai

# Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux / macOS
# or
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

---

## Configuration

```bash
# Copy environment configuration template
cp .env.example .env

# Edit API keys and database settings
nano .env

# Initialize HALDxAI configuration
python -m haldxai.cli init
```

---

## Quick Example

```python
from haldxai import HALDxAI

# Initialize the system
hald = HALDxAI(config_path="configs/config.yaml")

# Run NER pipeline
results = hald.run_ner_pipeline(
    input_file="data/raw/articles.csv",
    output_dir="data/processed/"
)

# Build the knowledge graph
graph = hald.build_knowledge_graph(results)

# Generate report
hald.generate_report(graph, output_path="reports/analysis.html")
```

---

# 📁 Project Structure

```
HALDxAI/
├── haldxai/               # Main Python package
│   ├── core/              # Core modules
│   ├── ner/               # NER pipelines
│   ├── database/          # Database interfaces
│   ├── modeling/          # ML/DL models
│   ├── scoring/           # Scoring system
│   ├── visualization/     # Visualization tools
│   └── workflow/          # End-to-end workflows
├── notebooks/             # Analysis notebooks
├── configs/               # Configurations
├── data/                  # Data directories
├── scripts/               # Utility scripts
├── tests/                 # Test suite
└── docs/                  # Documentation
```

---

# 📖 Documentation

* Installation Guide – `docs/installation.md`
* Usage Tutorial – `docs/usage.md`
* API Reference – `docs/api/`
* Example Workflows & Case Studies – `docs/examples/`

---

# 🧪 Running Tests

```bash
pytest tests/                # Run all tests
pytest tests/test_core/      # Run tests for core modules
pytest --cov=haldxai tests/  # Coverage report
```

---

# 🤝 Contributing

We welcome contributions from the community.

Please read the **CONTRIBUTING.md** for guidelines on:

* Submitting pull requests
* Development environment setup
* Code style (black, isort)
* Type checking with mypy
* Pre-commit hooks

### Development Setup

```bash
pip install -r requirements-dev.txt
pre-commit install
black haldxai/
isort haldxai/
mypy haldxai/
```

---

# 📄 License

This project is released under the **MIT License**.
See the `LICENSE` file for details.

---

# 📊 Citation

If you use HALDxAI in your research, please cite:

```bibtex
@software{haldxai2024,
  title={HALDxAI: Healthy Aging and Longevity Discovery AI},
  author={HALDxAI Development Team},
  year={2024},
  url={https://github.com/ShilongZhang116/haldxai}
}
```

---

# 🌐 Online Service (Demo)

A public demo of HALDxAI is available at:
[https://bis.zju.edu.cn/haldxai](https://bis.zju.edu.cn/haldxai)

---

# ⚠️ Disclaimer

HALDxAI is intended for academic research only and does not provide medical advice.
