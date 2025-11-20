# HALDxAI Usage Guide

This guide provides comprehensive instructions for using HALDxAI (Healthy Aging and Longevity Discovery AI Platform).

## Table of Contents

- [Quick Start](#quick-start)
- [Command Line Interface](#command-line-interface)
- [Python API](#python-api)
- [Configuration](#configuration)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Analysis and Visualization](#analysis-and-visualization)
- [Advanced Usage](#advanced-usage)
- [Best Practices](#best-practices)

## Quick Start

### Basic Workflow

1. **Initialize Project**
   ```bash
   haldxai init my_haldxai_project
   cd my_haldxai_project
   ```

2. **Configure Environment**
   ```bash
   # Edit .env file with your API keys
   nano .env
   ```

3. **Run Complete Pipeline**
   ```bash
   haldxai process data/articles.csv outputs/
   ```

4. **View Results**
   ```bash
   # Open generated report
   open outputs/report.html
   ```

### Example with Sample Data

```bash
# Download sample data
wget https://example.com/haldxai_sample_data.csv

# Run pipeline with sample data
haldxai process haldxai_sample_data.csv sample_results/
```

## Command Line Interface

### Main Commands

#### `haldxai init`
Initialize a new HALDxAI project.

```bash
haldxai init [project_path] [options]

Options:
  --force          Overwrite existing files
  --config PATH    Use custom configuration
```

#### `haldxai process`
Run the complete processing pipeline.

```bash
haldxai process [input_file] [options]

Options:
  --output, -o PATH     Output directory (default: outputs)
  --ner METHOD           NER method: llm or spacy
  --relations/--no-relations  Extract relations
  --scoring/--no-scoring    Run scoring
  --report/--no-report      Generate report
```

#### `haldxai analyze`
Analyze existing processed data.

```bash
haldxai analyze [data_dir] [options]

Options:
  --output, -o PATH     Analysis output directory
  --report-type TYPE     Report format: html or json
```

#### `haldxai config`
Manage configuration settings.

```bash
haldxai config [options]

Options:
  --show              Show current configuration
  --validate          Validate configuration
  --key KEY          Set configuration key
  --value VALUE      Set configuration value
```

### Advanced Commands

#### Pipeline Modes

```bash
# Run only NER
haldxai process input.csv --ner llm --no-relations --no-scoring

# Run with custom scoring
haldxai process input.csv --scoring-types aging_relevance bridge_candidates

# Analysis mode only
haldxai analyze processed_data/ --output analysis/
```

#### Configuration Management

```bash
# Show configuration
haldxai config --show

# Validate configuration
haldxai config --validate

# Set configuration values
haldxai config --key processing.max_workers --value 8
```

## Python API

### Basic Usage

```python
from haldxai import HALDxAI

# Initialize with default configuration
hald = HALDxAI()

# Initialize with custom configuration
hald = HALDxAI(config_path="configs/my_config.yaml")
```

### Data Processing

#### Named Entity Recognition

```python
# Run NER pipeline
results = hald.run_ner_pipeline(
    input_file="data/articles.csv",
    output_dir="outputs/ner/",
    ner_method="llm"
)

if results.success:
    print(f"NER completed: {results.message}")
    print(f"Found {len(results.data.get('entities', []))} entities")
else:
    print(f"NER failed: {results.message}")
```

#### Relation Extraction

```python
# Run relation extraction
results = hald.run_relation_extraction(
    entities_file="outputs/ner/entities.json",
    output_dir="outputs/relations/"
)

if results.success:
    print(f"Relation extraction completed: {results.message}")
    print(f"Found {len(results.data.get('relations', []))} relations")
```

#### Scoring

```python
# Run scoring pipeline
results = hald.run_scoring_pipeline(
    output_dir="outputs/scoring/",
    scoring_types=["aging_relevance", "bridge_candidates"]
)

if results.success:
    print(f"Scoring completed: {results.message}")
    print(f"Scores: {results.data}")
```

### Knowledge Graph Construction

```python
# Build knowledge graph
results = hald.build_knowledge_graph(
    entities_file="outputs/entities.json",
    relations_file="outputs/relations.json",
    output_dir="outputs/graph/"
)

if results.success:
    print(f"Knowledge graph built: {results.message}")
    graph_data = results.data
    print(f"Nodes: {len(graph_data['nodes'])}")
    print(f"Edges: {len(graph_data['edges'])}")
```

### Report Generation

```python
# Generate HTML report
results = hald.generate_report(
    output_path="outputs/report.html",
    report_type="html",
    include_visualizations=True
)

if results.success:
    print(f"Report generated: {results.message}")
```

### Data Management

#### Saving Data

```python
# Save all processed data
results = hald.save_data("outputs/processed_data/")

if results.success:
    print(f"Data saved: {results.message}")
```

#### Loading Data

```python
# Load existing data
results = hald.load_data("outputs/processed_data/")

if results.success:
    print(f"Data loaded: {results.message}")
    
    # Access managers
    entities = hald.entity_manager.get_all_entities()
    relations = hald.relation_manager.get_all_relations()
```

## Configuration

### Configuration Structure

HALDxAI uses a hierarchical configuration system:

```yaml
# Main configuration file: configs/config.yaml
paths:
  project_root: "."
  data_dir: "data"
  output_dir: "outputs"

database:
  postgresql:
    host: "localhost"
    port: 5432
    dbname: "haldxai"
  neo4j:
    uri: "bolt://localhost:7687"

api:
  deepseek:
    api_key: "${DEEPSEEK_API_KEY}"
    model: "deepseek-chat"
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-3.5-turbo"

processing:
  max_workers: 16
  chunk_size: 200
  llm_temperature: 0.1
```

### Environment Variables

Key environment variables:

```bash
# API Keys
export DEEPSEEK_API_KEY="your_deepseek_key"
export OPENAI_API_KEY="your_openai_key"
export PUBMED_API_KEY="your_pubmed_key"

# Database
export PG_HOST="localhost"
export PG_USER="postgres"
export PG_PASS="your_password"

# Processing
export MAX_WORKERS="8"
export LOG_LEVEL="DEBUG"
```

### Custom Configuration

```python
from haldxai.config import Settings, load_config, save_config

# Load configuration
settings = load_config("configs/config.yaml")

# Modify settings
settings.processing.max_workers = 8
settings.api.deepseek.model = "deepseek-coder"

# Save configuration
save_config(settings, "configs/custom_config.yaml")
```

## Data Processing Pipeline

### Pipeline Stages

1. **Data Input**: Load and validate input data
2. **Named Entity Recognition**: Extract aging-related entities
3. **Relation Extraction**: Identify relationships between entities
4. **Scoring**: Calculate relevance and importance scores
5. **Knowledge Graph**: Build graph structure
6. **Report Generation**: Create analysis reports

### Custom Pipeline

```python
from haldxai.core.pipeline import HALDxAI, PipelineConfig

# Create custom pipeline configuration
pipeline_config = PipelineConfig(
    config_path="configs/custom.yaml",
    data_dir="data/custom/",
    output_dir="outputs/custom/",
    log_level="DEBUG",
    parallel_processing=True,
    max_workers=8
)

# Initialize with custom config
hald = HALDxAI(pipeline_config)
```

### Batch Processing

```python
# Process multiple files
import glob
from pathlib import Path

input_files = list(Path("data/").glob("*.csv"))

for i, file_path in enumerate(input_files):
    print(f"Processing file {i+1}/{len(input_files)}: {file_path}")
    
    output_dir = f"outputs/batch_{i+1}/"
    results = hald.run_ner_pipeline(file_path, output_dir)
    
    if not results.success:
        print(f"Failed to process {file_path}: {results.message}")
```

## Analysis and Visualization

### Entity Analysis

```python
# Get entity statistics
stats = hald.entity_manager.get_statistics()
print(f"Total entities: {stats['total_entities']}")
print(f"Type distribution: {stats['type_counts']}")

# Search entities
aging_entities = hald.entity_manager.get_entities_by_type(
    haldxai.core.types.EntityType.AAI
)
print(f"Found {len(aging_entities)} anti-aging interventions")
```

### Relation Analysis

```python
# Get relation statistics
stats = hald.relation_manager.get_statistics()
print(f"Total relations: {stats['total_relations']}")
print(f"Directed relations: {stats['directed_relations']}")

# Get relations between specific entities
relations = hald.relation_manager.get_relations_between_entities(
    "entity_1", "entity_2"
)
print(f"Found {len(relations)} relations between entities")
```

### Network Visualization

```python
from haldxai.visualization import NetworkVisualizer

# Create visualizer
visualizer = NetworkVisualizer(hald.config)

# Generate network plot
network_data = hald.relation_manager.get_network_data()
viz_results = visualizer.create_network_plot(
    network_data,
    output_path="outputs/network.png",
    layout="spring",
    node_size_metric="confidence"
)

if viz_results.success:
    print(f"Network visualization saved: {viz_results.message}")
```

## Advanced Usage

### Custom Entity Types

```python
from haldxai.core.types import EntityType

# Define custom entity type
class CustomEntityType(EntityType):
    CUSTOM_PROTEIN = "CUSTOM_PROTEIN"
    CUSTOM_PATHWAY = "CUSTOM_PATHWAY"

# Use in processing
hald.run_ner_pipeline(
    input_file="data.csv",
    output_dir="outputs/",
    custom_types=[CustomEntityType.CUSTOM_PROTEIN]
)
```

### Custom Scoring Functions

```python
from haldxai.scoring import AgingScorer

# Create custom scorer
class CustomAgingScorer(AgingScorer):
    def score_entities(self, entities):
        # Custom scoring logic
        scores = {}
        for entity in entities:
            score = self.calculate_custom_score(entity)
            scores[entity.id] = score
        return scores
    
    def calculate_custom_score(self, entity):
        # Your custom scoring algorithm
        return 0.85  # Example score

# Use custom scorer
scorer = CustomAgingScorer(hald.config)
results = scorer.score_entities(entities)
```

### Integration with External Tools

```python
# Integrate with pandas
import pandas as pd

# Load HALDxAI results into DataFrame
entities_df = pd.DataFrame([
    {
        'id': entity.id,
        'name': entity.name,
        'type': entity.entity_type.value,
        'confidence': entity.confidence
    }
    for entity in hald.entity_manager.get_all_entities()
])

# Perform analysis
entity_counts = entities_df['type'].value_counts()
print(entity_counts)

# Integrate with networkx
import networkx as nx

# Create NetworkX graph
G = nx.Graph()

# Add nodes
for entity in hald.entity_manager.get_all_entities():
    G.add_node(entity.id, **entity.to_dict())

# Add edges
for relation in hald.relation_manager.get_all_relations():
    G.add_edge(relation.source_id, relation.target_id, **relation.to_dict())

# Perform network analysis
centrality = nx.betweenness_centrality(G)
print(f"Node centrality: {centrality}")
```

## Best Practices

### Performance Optimization

1. **Batch Processing**
   ```python
   # Use appropriate batch sizes
   hald.run_ner_pipeline(
       input_file="large_dataset.csv",
       output_dir="outputs/",
       batch_size=1000  # Adjust based on available memory
   )
   ```

2. **Parallel Processing**
   ```yaml
   # In config.yaml
   processing:
     parallel_processing: true
     max_workers: 8  # Set to number of CPU cores
   ```

3. **Caching**
   ```yaml
   # Enable caching for repeated operations
   processing:
     enable_caching: true
     cache_ttl: 3600  # Cache for 1 hour
   ```

### Data Quality

1. **Input Validation**
   ```python
   # Validate input data before processing
   import pandas as pd
   
   df = pd.read_csv("input.csv")
   
   # Check required columns
   required_columns = ['pmid', 'title', 'abstract']
   missing_columns = [col for col in required_columns if col not in df.columns]
   
   if missing_columns:
       raise ValueError(f"Missing required columns: {missing_columns}")
   ```

2. **Quality Filtering**
   ```python
   # Filter low-confidence entities
   high_confidence_entities = [
       entity for entity in hald.entity_manager.get_all_entities()
       if entity.confidence >= 0.8
   ]
   ```

### Error Handling

1. **Graceful Degradation**
   ```python
   try:
       results = hald.run_ner_pipeline(input_file, output_dir)
   except Exception as e:
       logger.error(f"NER pipeline failed: {e}")
       
       # Fallback to alternative method
       results = hald.run_ner_pipeline(input_file, output_dir, ner_method="spacy")
   ```

2. **Retry Logic**
   ```python
   from haldxai.utils import retry_with_backoff
   
   @retry_with_backoff(max_retries=3, initial_delay=1.0)
   def call_api_with_retry():
       # API call that might fail
       return api_call()
   ```

### Resource Management

1. **Memory Monitoring**
   ```python
   import psutil
   
   # Monitor memory usage
   memory_percent = psutil.virtual_memory().percent
   if memory_percent > 80:
       logger.warning("High memory usage detected")
       # Reduce batch size or enable chunking
   ```

2. **Progress Tracking**
   ```python
   from haldxai.utils.logging import ProgressLogger
   
   # Track progress for long operations
   progress = ProgressLogger(logger, total_items=1000, description="Processing entities")
   
   for i, item in enumerate(items):
       # Process item
       process_item(item)
       progress.update()
   
   progress.finish("Completed processing")
   ```

## Troubleshooting

### Common Issues

#### Memory Errors
```python
# Reduce memory usage
hald.run_ner_pipeline(
    input_file="large_file.csv",
    output_dir="outputs/",
    chunk_size=500,  # Reduce from default
    max_workers=2     # Reduce parallelism
)
```

#### API Rate Limits
```python
# Configure retry with backoff
from haldxai.utils import retry_with_backoff

@retry_with_backoff(max_retries=5, initial_delay=2.0, max_delay=60.0)
def robust_api_call():
    return api_client.process(text)
```

#### Database Issues
```python
# Test database connection
from haldxai.database import PostgreSQLManager

try:
    db_manager = PostgreSQLManager(hald.config)
    db_manager.test_connection()
    print("Database connection successful")
except Exception as e:
    print(f"Database connection failed: {e}")
    # Check configuration and database status
```

### Getting Help

For additional support:

1. **Documentation**: https://haldxai.readthedocs.io
2. **Issues**: https://github.com/your-org/HALDxAI-Repository/issues
3. **Discussions**: https://github.com/your-org/HALDxAI-Repository/discussions
4. **Email**: haldxai@example.com

---

This guide covers the main usage patterns for HALDxAI. For specific use cases and advanced configurations, see the [API Documentation](api/) and explore the example notebooks in the `notebooks/` directory.