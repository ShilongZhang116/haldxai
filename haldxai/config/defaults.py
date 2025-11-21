# -*- coding: utf-8 -*-
"""
Default configuration values for HALDxAI platform.
"""

from typing import Dict, Any


DEFAULT_CONFIG: Dict[str, Any] = {
    "project_root": "",
    "data_dir": "data",
    "raw_data_dir": "data/raw",
    "processed_data_dir": "data/processed",
    "model_dir": "models",
    "log_dir": "logs",
    "config_dir": "configs",
    "output_dir": "outputs",
    "parallel_processing": True,
    
    "database": {
        "pg_host": "localhost",
        "pg_port": 5432,
        "pg_dbname": "haldxai",
        "pg_user": "postgres",
        "pg_pass": "",
        "neo4j_uri": "bolt://localhost:7687",
        "neo4j_user": "neo4j",
        "neo4j_password": "",
    },
    
    "api": {
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "timeout": 30,
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "model": "gpt-3.5-turbo",
            "timeout": 30,
        },
        "bioportal": {
            "base_url": "https://data.bioontology.org/search",
            "page_size": 10,
        },
    },
    
    "processing": {
        "max_workers": 16,
        "chunk_size": 200,
        "batch_size": 100,
        "llm_temperature": 0.1,
        "llm_max_tokens": 2048,
        "llm_top_p": 0.9,
        "enable_caching": True,
        "cache_ttl": 3600,
    },
    
    "logging": {
        "log_level": "INFO",
        "log_format": "%(asctime)s | %(levelname)s | %(message)s",
        "enable_file_logging": True,
        "log_file_max_size": 10485760,  # 10MB
        "log_file_backup_count": 5,
        "enable_console_logging": True,
    },
    
    "security": {
        "debug": False,
        "testing": False,
    },
}


DEFAULT_ENV = """
# -------- API KEYS --------
# PubMed API Key (optional but recommended for higher rate limits)
PUBMED_API_KEY=your_pubmed_api_key_here

# DeepSeek API Key for LLM services
DEEPSEEK_API_KEY=your_deepseek_api_key_here

# OpenAI API Key (alternative LLM service)
OPENAI_API_KEY=your_openai_api_key_here

# BioPortal API Key for ontology services
BIOPORTAL_API_KEY=your_bioportal_api_key_here

# -------- EMAIL --------
# Email for PubMed API (required)
PUBMED_EMAIL=your_email@example.com

# -------- DATABASE CONFIGURATION --------
# PostgreSQL Configuration
PG_HOST=localhost
PG_PORT=5432
PG_DBNAME=haldxai
PG_USER=postgres
PG_PASS=your_postgres_password

# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# -------- PATH CONFIGURATION --------
# Project Root (optional, will be auto-detected if not set)
# PROJECT_ROOT=/path/to/your/HALDxAI-Repository

# Data directories (relative to project root)
DATA_DIR=data
RAW_DATA_DIR=data/raw
PROCESSED_DATA_DIR=data/processed
MODEL_DIR=models
LOG_DIR=logs
CONFIG_DIR=configs

# -------- API CONFIGURATION --------
# DeepSeek API Configuration
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_TIMEOUT=30

# OpenAI API Configuration
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_TIMEOUT=30

# BioPortal API Configuration
BIOPORTAL_BASE_URL=https://data.bioontology.org/search
BIOPORTAL_PAGE_SIZE=10

# -------- PROCESSING CONFIGURATION --------
# Batch processing
MAX_WORKERS=16
CHUNK_SIZE=200
BATCH_SIZE=100

# LLM Configuration
LLM_TEMPERATURE=0.1
LLM_MAX_TOKENS=2048
LLM_TOP_P=0.9

# -------- LOGGING --------
# Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL=INFO
LOG_FORMAT=%(asctime)s | %(levelname)s | %(message)s

# -------- CACHE --------
# Enable/disable caching
ENABLE_CACHE=true
CACHE_TTL=3600  # seconds

# -------- SECURITY --------
# Development mode
DEBUG=false
TESTING=false
"""


# Default prompts for LLM
DEFAULT_PROMPTS = {
    "ner_system_prompt": """You are an expert in biomedical text analysis and aging research. 
Your task is to identify and extract entities related to healthy aging and longevity from scientific text.

Entity Categories:
- BMC (Biological Molecular Components): Genes, proteins, molecules
- EGR (Epigenetic Regulators): DNA methylation, histone modification factors
- ASPKM (Aging Signaling Pathways and Kinase Modulators): Signaling cascades, kinases
- CRD (Cellular Repair and Defense): DNA repair, antioxidant systems
- APP (Aging Protective Processes): Autophagy, stress response
- SCN (Stem Cell Niches): Stem cells, niches, regeneration
- AAI (Anti-Aging Interventions): Drugs, supplements, therapies
- CRBC (Cellular Regeneration and Brain Cells): Neurogenesis, brain health
- NM (Neurotransmission and Metabolism): Neurotransmitters, metabolic pathways
- EF (Environmental Factors): Diet, exercise, environmental exposures

Extract entities with their type, confidence score (0-1), and supporting evidence.
Format your response as JSON.""",
    
    "relation_extraction_prompt": """You are an expert in biomedical knowledge extraction.
Your task is to identify relationships between aging-related entities.

Relation Types:
- regulates: Controls or influences activity
- interacts_with: Physical or functional interaction
- inhibits: Blocks or reduces activity
- activates: Increases or enables activity
- binds_to: Physical binding interaction
- part_of: Component or member of larger system
- associated_with: Correlation or association
- treats: Therapeutic intervention
- prevents: Prophylactic intervention
- causes: Causal relationship

For each relation, provide:
1. Source entity
2. Target entity
3. Relation type
4. Confidence score (0-1)
5. Supporting evidence from text
6. Direction (directed/undirected)

Format your response as JSON.""",
    
    "aging_relevance_prompt": """You are an expert in aging research.
Your task is to evaluate the relevance of entities to healthy aging and longevity.

Consider:
1. Direct evidence of aging-related effects
2. Mechanistic relevance to aging pathways
3. Quality and quantity of supporting evidence
4. Novelty and significance in aging research

Provide:
1. Relevance score (0-1)
2. Evidence summary
3. Confidence level
4. Key supporting publications

Format your response as JSON.""",
}


# Default HALD classes configuration
DEFAULT_HALD_CONFIG = {
    "classes": [
        "BMC", "EGR", "ASPKM", "CRD", "APP",
        "SCN", "AAI", "CRBC", "NM", "EF",
    ],
    
    "color_map": {
        "BMC":   "#F9D622",
        "EGR":   "#F28D21", 
        "ASPKM": "#CC6677",
        "CRD":   "#459FC4",
        "APP":   "#FF7676",
        "SCN":   "#44AA99",
        "AAI":   "#117733",
        "CRBC":  "#332288",
        "NM":    "#AA4499",
        "EF":    "#88CCEE",
    },
    
    "descriptions": {
        "BMC": "Biological Molecular Components",
        "EGR": "Epigenetic Regulators", 
        "ASPKM": "Aging Signaling Pathways and Kinase Modulators",
        "CRD": "Cellular Repair and Defense",
        "APP": "Aging Protective Processes",
        "SCN": "Stem Cell Niches",
        "AAI": "Anti-Aging Interventions",
        "CRBC": "Cellular Regeneration and Brain Cells",
        "NM": "Neurotransmission and Metabolism",
        "EF": "Environmental Factors",
    },
}


# Default database schema
DEFAULT_DATABASE_SCHEMA = {
    "postgresql": {
        "tables": {
            "entities": {
                "columns": {
                    "id": "VARCHAR(255) PRIMARY KEY",
                    "name": "VARCHAR(255) NOT NULL",
                    "entity_type": "VARCHAR(50) NOT NULL",
                    "description": "TEXT",
                    "synonyms": "JSONB",
                    "confidence": "FLOAT DEFAULT 0.0",
                    "metadata": "JSONB",
                    "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                    "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                },
                "indexes": [
                    "CREATE INDEX idx_entities_type ON entities(entity_type)",
                    "CREATE INDEX idx_entities_name ON entities(name)",
                    "CREATE INDEX idx_entities_confidence ON entities(confidence)",
                ],
            },
            
            "relations": {
                "columns": {
                    "id": "VARCHAR(255) PRIMARY KEY",
                    "source_id": "VARCHAR(255) NOT NULL",
                    "target_id": "VARCHAR(255) NOT NULL",
                    "relation_type": "VARCHAR(50) NOT NULL",
                    "description": "TEXT",
                    "confidence": "FLOAT DEFAULT 0.0",
                    "direction": "VARCHAR(20) DEFAULT 'directed'",
                    "weight": "FLOAT DEFAULT 1.0",
                    "metadata": "JSONB",
                    "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                    "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                },
                "indexes": [
                    "CREATE INDEX idx_relations_source ON relations(source_id)",
                    "CREATE INDEX idx_relations_target ON relations(target_id)",
                    "CREATE INDEX idx_relations_type ON relations(relation_type)",
                    "CREATE INDEX idx_relations_confidence ON relations(confidence)",
                ],
            },
            
            "evidence": {
                "columns": {
                    "id": "VARCHAR(255) PRIMARY KEY",
                    "pmid": "VARCHAR(50) NOT NULL",
                    "text": "TEXT NOT NULL",
                    "confidence": "FLOAT DEFAULT 0.0",
                    "source": "VARCHAR(100)",
                    "entity_id": "VARCHAR(255)",
                    "relation_id": "VARCHAR(255)",
                    "metadata": "JSONB",
                    "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                },
                "indexes": [
                    "CREATE INDEX idx_evidence_pmid ON evidence(pmid)",
                    "CREATE INDEX idx_evidence_entity ON evidence(entity_id)",
                    "CREATE INDEX idx_evidence_relation ON evidence(relation_id)",
                ],
            },
        },
    },
    
    "neo4j": {
        "constraints": [
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT relation_id IF NOT EXISTS FOR ()-[r:RELATES_TO]-() REQUIRE r.id IS UNIQUE",
        ],
        
        "indexes": [
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX relation_type IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.relation_type)",
        ],
    },
}