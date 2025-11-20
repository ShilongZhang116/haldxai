# HALDxAI Installation Guide

This guide will help you install and set up HALDxAI (Healthy Aging and Longevity Discovery AI Platform) on your system.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation Methods](#installation-methods)
- [Configuration](#configuration)
- [Database Setup](#database-setup)
- [Verification](#verification)
- [Troubleshooting](#troubleshooting)

## System Requirements

### Minimum Requirements

- **Operating System**: Linux, macOS, or Windows (64-bit)
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM (16GB+ recommended for large datasets)
- **Storage**: 10GB free disk space
- **Network**: Internet connection for API access

### Recommended Requirements

- **Operating System**: Ubuntu 20.04+ / macOS 12+ / Windows 10+
- **Python**: 3.9 or higher
- **Memory**: 16GB+ RAM
- **Storage**: 50GB+ free disk space
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for accelerated processing)

### Optional Dependencies

- **PostgreSQL**: 12+ (for database functionality)
- **Neo4j**: 4.0+ (for graph database functionality)
- **Docker**: 20.10+ (for containerized deployment)

## Installation Methods

### Method 1: Standard Installation (Recommended)

1. **Clone the Repository**
   ```bash
   git clone https://github.com/ShilongZhang116/haldxai.git
   cd haldxai
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # or
   venv\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Install HALDxAI**
   ```bash
   pip install -e .
   ```

### Method 2: Development Installation

For developers who want to modify the code:

1. **Clone with Development Branch**
   ```bash
   git clone -b develop https://github.com/ShilongZhang116/haldxai.git
   cd HALDxAI-Repository
   ```

2. **Install Development Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e ".[dev]"
   ```

3. **Setup Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

### Method 3: Conda Installation

1. **Create Conda Environment**
   ```bash
   conda create -n haldxai python=3.9
   conda activate haldxai
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit the `.env` file with your settings:

```bash
# API Keys
PUBMED_API_KEY=your_pubmed_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
OPENAI_API_KEY=your_openai_api_key
BIOPORTAL_API_KEY=your_bioportal_api_key

# Email for PubMed
PUBMED_EMAIL=your_email@example.com

# Database Configuration
PG_HOST=localhost
PG_PORT=5432
PG_DBNAME=haldxai
PG_USER=postgres
PG_PASS=your_password

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
```

### 2. Configuration File

The main configuration is in `configs/config.yaml`. You can customize:

```bash
# Copy default configuration
cp configs/config.yaml configs/config.yaml.local
```

Key configuration sections:
- `paths`: Directory paths
- `database`: Database connection settings
- `api`: API configuration
- `processing`: Processing parameters
- `logging`: Log settings

### 3. Project Initialization

Initialize a new HALDxAI project:

```bash
haldxai init /path/to/your/project
```

This creates the directory structure and configuration files.

## Database Setup

### PostgreSQL Setup

1. **Install PostgreSQL**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install postgresql postgresql-contrib
   
   # macOS
   brew install postgresql
   
   # Windows
   # Download and install from postgresql.org
   ```

2. **Create Database and User**
   ```bash
   sudo -u postgres psql
   CREATE DATABASE haldxai;
   CREATE USER haldxai_user WITH PASSWORD 'your_password';
   GRANT ALL PRIVILEGES ON DATABASE haldxai TO haldxai_user;
   \q
   ```

3. **Setup Database Schema**
   ```bash
   python scripts/setup_database.py --config configs/config.yaml
   ```

### Neo4j Setup

1. **Install Neo4j**
   ```bash
   # Download from neo4j.com
   # or use Docker
   docker run --rm -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
   ```

2. **Configure Neo4j**
   Edit `conf/neo4j.conf` to enable remote connections.

3. **Setup HALDxAI Schema**
   ```bash
   python scripts/setup_database.py --config configs/config.yaml --neo4j-only
   ```

## Verification

### 1. Test Installation

Verify the installation:

```bash
haldxai --version
```

Expected output:
```
HALDxAI version 0.1.0
```

### 2. Test Configuration

Check your configuration:

```bash
haldxai config --show
```

### 3. Test Database Connection

Test database connectivity:

```bash
haldxai config --validate
```

### 4. Run Basic Pipeline

Test with sample data:

```bash
# Create test data directory
mkdir -p test_data
echo "Sample text about aging research..." > test_data/sample.txt

# Run NER pipeline
haldxai process test_data/sample.txt test_output --ner llm --relations --no-scoring --no-report
```

## Troubleshooting

### Common Issues

#### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'haldxai'`

**Solution**:
1. Ensure you're in the correct virtual environment
2. Install in development mode: `pip install -e .`
3. Check Python path: `echo $PYTHONPATH`

#### Database Connection Errors

**Problem**: `Connection refused` or `authentication failed`

**Solution**:
1. Verify database services are running
2. Check connection parameters in `.env`
3. Ensure firewall allows database connections
4. Verify user permissions

#### Memory Issues

**Problem**: `MemoryError` or system becomes unresponsive

**Solution**:
1. Reduce batch size in configuration
2. Use chunked processing
3. Increase system RAM or use cloud resources
4. Enable memory optimization in config

#### API Rate Limiting

**Problem**: API rate limit exceeded

**Solution**:
1. Add API keys to increase limits
2. Implement retry logic with backoff
3. Use batch processing
4. Consider premium API plans

### Performance Optimization

#### Memory Optimization

```yaml
# In configs/config.yaml
processing:
  chunk_size: 1000  # Reduce from default
  max_workers: 4     # Reduce parallelism
  enable_caching: true
```

#### GPU Acceleration

```bash
# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### Database Optimization

```sql
-- PostgreSQL optimization
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
```

### Getting Help

If you encounter issues:

1. **Check Logs**: Look in `logs/haldxai.log`
2. **Verbose Mode**: Run with `--verbose` flag
3. **Community Support**: Open an issue on GitHub
4. **Documentation**: Check the latest docs at https://haldxai.readthedocs.io

### Known Limitations

- **Large Datasets**: May require significant RAM
- **API Limits**: Rate limiting affects processing speed
- **GPU Memory**: Large models may not fit on consumer GPUs
- **Database Size**: Neo4j performance degrades with very large graphs

## Next Steps

After successful installation:

1. **Review Documentation**: Read the [Usage Guide](usage.md)
2. **Explore Notebooks**: Check `notebooks/` directory for examples
3. **Join Community**: Participate in discussions and contributions
4. **Stay Updated**: Watch for new releases and features

---

For additional help or questions, please:
- Open an issue on [GitHub Issues](https://github.com/ShilongZhang116/haldxai/issues)
- Email us at shilongzhang@zju.edu.cn