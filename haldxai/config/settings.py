# -*- coding: utf-8 -*-
"""
Settings and configuration management for HALDxAI platform.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class DatabaseConfig:
    """Database configuration."""
    # PostgreSQL
    pg_host: str = field(default_factory=lambda: os.getenv("PG_HOST", "localhost"))
    pg_port: int = field(default_factory=lambda: int(os.getenv("PG_PORT", "5432")))
    pg_dbname: str = field(default_factory=lambda: os.getenv("PG_DBNAME", "haldxai"))
    pg_user: str = field(default_factory=lambda: os.getenv("PG_USER", "postgres"))
    pg_pass: str = field(default_factory=lambda: os.getenv("PG_PASS", ""))
    
    # Neo4j
    neo4j_uri: str = field(default_factory=lambda: os.getenv("NEO4J_URI", "bolt://localhost:7687"))
    neo4j_user: str = field(default_factory=lambda: os.getenv("NEO4J_USER", "neo4j"))
    neo4j_password: str = field(default_factory=lambda: os.getenv("NEO4J_PASSWORD", ""))


@dataclass
class APIConfig:
    """API configuration."""
    # DeepSeek
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = field(default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    deepseek_model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-chat"))
    deepseek_timeout: int = field(default_factory=lambda: int(os.getenv("DEEPSEEK_TIMEOUT", "30")))
    
    # OpenAI
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_base_url: str = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"))
    openai_timeout: int = field(default_factory=lambda: int(os.getenv("OPENAI_TIMEOUT", "30")))
    
    # PubMed
    pubmed_api_key: str = field(default_factory=lambda: os.getenv("PUBMED_API_KEY", ""))
    pubmed_email: str = field(default_factory=lambda: os.getenv("PUBMED_EMAIL", ""))
    
    # BioPortal
    bioportal_api_key: str = field(default_factory=lambda: os.getenv("BIOPORTAL_API_KEY", ""))
    bioportal_base_url: str = field(default_factory=lambda: os.getenv("BIOPORTAL_BASE_URL", "https://data.bioontology.org/search"))
    bioportal_page_size: int = field(default_factory=lambda: int(os.getenv("BIOPORTAL_PAGE_SIZE", "10")))


@dataclass
class ProcessingConfig:
    """Processing configuration."""
    max_workers: int = field(default_factory=lambda: int(os.getenv("MAX_WORKERS", "16")))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "200")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "100")))
    
    # LLM parameters
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.1")))
    llm_max_tokens: int = field(default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2048")))
    llm_top_p: float = field(default_factory=lambda: float(os.getenv("LLM_TOP_P", "0.9")))
    
    # Caching
    enable_caching: bool = field(default_factory=lambda: os.getenv("ENABLE_CACHE", "true").lower() == "true")
    cache_ttl: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL", "3600")))


@dataclass
class PathConfig:
    """Path configuration."""
    project_root: str = field(default_factory=lambda: os.getenv("PROJECT_ROOT", str(Path.cwd())))
    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "data"))
    raw_data_dir: str = field(default_factory=lambda: os.getenv("RAW_DATA_DIR", "data/raw"))
    processed_data_dir: str = field(default_factory=lambda: os.getenv("PROCESSED_DATA_DIR", "data/processed"))
    model_dir: str = field(default_factory=lambda: os.getenv("MODEL_DIR", "models"))
    log_dir: str = field(default_factory=lambda: os.getenv("LOG_DIR", "logs"))
    config_dir: str = field(default_factory=lambda: os.getenv("CONFIG_DIR", "configs"))
    
    def get_absolute_path(self, relative_path: str) -> Path:
        """Get absolute path from relative path."""
        base_path = Path(self.project_root)
        return base_path / relative_path
    
    def get_data_path(self) -> Path:
        """Get data directory path."""
        return self.get_absolute_path(self.data_dir)
    
    def get_raw_data_path(self) -> Path:
        """Get raw data directory path."""
        return self.get_absolute_path(self.raw_data_dir)
    
    def get_processed_data_path(self) -> Path:
        """Get processed data directory path."""
        return self.get_absolute_path(self.processed_data_dir)
    
    def get_model_path(self) -> Path:
        """Get model directory path."""
        return self.get_absolute_path(self.model_dir)
    
    def get_log_path(self) -> Path:
        """Get log directory path."""
        return self.get_absolute_path(self.log_dir)
    
    def get_config_path(self) -> Path:
        """Get config directory path."""
        return self.get_absolute_path(self.config_dir)


@dataclass
class LoggingConfig:
    """Logging configuration."""
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_format: str = field(default_factory=lambda: os.getenv("LOG_FORMAT", "%(asctime)s | %(levelname)s | %(message)s"))
    
    # File logging
    enable_file_logging: bool = True
    log_file_max_size: int = 10 * 1024 * 1024  # 10MB
    log_file_backup_count: int = 5
    
    # Console logging
    enable_console_logging: bool = True


@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "your-secret-key-here"))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    testing: bool = field(default_factory=lambda: os.getenv("TESTING", "false").lower() == "true")


@dataclass
class Settings:
    """Main settings class for HALDxAI platform."""
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    
    # Additional settings
    output_dir: str = "outputs"
    parallel_processing: bool = True
    
    def __post_init__(self):
        """Post-initialization setup."""
        # Ensure directories exist
        self.paths.get_data_path().mkdir(parents=True, exist_ok=True)
        self.paths.get_raw_data_path().mkdir(parents=True, exist_ok=True)
        self.paths.get_processed_data_path().mkdir(parents=True, exist_ok=True)
        self.paths.get_model_path().mkdir(parents=True, exist_ok=True)
        self.paths.get_log_path().mkdir(parents=True, exist_ok=True)
        self.paths.get_config_path().mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "database": {
                "pg_host": self.database.pg_host,
                "pg_port": self.database.pg_port,
                "pg_dbname": self.database.pg_dbname,
                "pg_user": self.database.pg_user,
                "pg_pass": "***" if self.database.pg_pass else None,
                "neo4j_uri": self.database.neo4j_uri,
                "neo4j_user": self.database.neo4j_user,
                "neo4j_password": "***" if self.database.neo4j_password else None,
            },
            "api": {
                "deepseek_base_url": self.api.deepseek_base_url,
                "deepseek_model": self.api.deepseek_model,
                "deepseek_timeout": self.api.deepseek_timeout,
                "deepseek_api_key": "***" if self.api.deepseek_api_key else None,
                "openai_base_url": self.api.openai_base_url,
                "openai_model": self.api.openai_model,
                "openai_timeout": self.api.openai_timeout,
                "openai_api_key": "***" if self.api.openai_api_key else None,
                "pubmed_api_key": "***" if self.api.pubmed_api_key else None,
                "pubmed_email": self.api.pubmed_email,
                "bioportal_base_url": self.api.bioportal_base_url,
                "bioportal_page_size": self.api.bioportal_page_size,
                "bioportal_api_key": "***" if self.api.bioportal_api_key else None,
            },
            "processing": {
                "max_workers": self.processing.max_workers,
                "chunk_size": self.processing.chunk_size,
                "batch_size": self.processing.batch_size,
                "llm_temperature": self.processing.llm_temperature,
                "llm_max_tokens": self.processing.llm_max_tokens,
                "llm_top_p": self.processing.llm_top_p,
                "enable_caching": self.processing.enable_caching,
                "cache_ttl": self.processing.cache_ttl,
            },
            "paths": {
                "project_root": self.paths.project_root,
                "data_dir": self.paths.data_dir,
                "raw_data_dir": self.paths.raw_data_dir,
                "processed_data_dir": self.paths.processed_data_dir,
                "model_dir": self.paths.model_dir,
                "log_dir": self.paths.log_dir,
                "config_dir": self.paths.config_dir,
            },
            "logging": {
                "log_level": self.logging.log_level,
                "log_format": self.logging.log_format,
                "enable_file_logging": self.logging.enable_file_logging,
                "log_file_max_size": self.logging.log_file_max_size,
                "log_file_backup_count": self.logging.log_file_backup_count,
                "enable_console_logging": self.logging.enable_console_logging,
            },
            "security": {
                "debug": self.security.debug,
                "testing": self.security.testing,
            },
            "output_dir": self.output_dir,
            "parallel_processing": self.parallel_processing,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Settings':
        """Create Settings from dictionary."""
        settings = cls()
        
        if "database" in data:
            db_config = data["database"]
            settings.database = DatabaseConfig(**db_config)
        
        if "api" in data:
            api_config = data["api"]
            settings.api = APIConfig(**api_config)
        
        if "processing" in data:
            proc_config = data["processing"]
            settings.processing = ProcessingConfig(**proc_config)
        
        if "paths" in data:
            path_config = data["paths"]
            settings.paths = PathConfig(**path_config)
        
        if "logging" in data:
            log_config = data["logging"]
            settings.logging = LoggingConfig(**log_config)
        
        if "security" in data:
            sec_config = data["security"]
            settings.security = SecurityConfig(**sec_config)
        
        # Update simple fields
        for field_name in ["output_dir", "parallel_processing"]:
            if field_name in data:
                setattr(settings, field_name, data[field_name])
        
        return settings


def load_config(config_path: Optional[Union[str, Path]] = None) -> Settings:
    """Load configuration from file or environment variables."""
    if config_path:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            return Settings.from_dict(config_data)
    
    # Return default settings
    return Settings()


def save_config(settings: Settings, config_path: Union[str, Path]) -> None:
    """Save configuration to file."""
    config_file = Path(config_path)
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, 'w', encoding='utf-8') as f:
        yaml.dump(settings.to_dict(), f, default_flow_style=False, allow_unicode=True)


def get_project_root() -> Path:
    """Get project root directory."""
    # Try to find .git directory or pyproject.toml
    current = Path.cwd()
    
    for _ in range(10):  # Limit search depth
        if (current / ".git").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent
    
    # Fallback to current directory
    return Path.cwd()


def validate_config(settings: Settings) -> list[str]:
    """Validate configuration and return list of issues."""
    issues = []
    
    # Check database configuration
    if not settings.database.pg_host:
        issues.append("PostgreSQL host is required")
    if not settings.database.pg_user:
        issues.append("PostgreSQL user is required")
    if not settings.database.pg_dbname:
        issues.append("PostgreSQL database name is required")
    
    # Check API configuration
    if not settings.api.deepseek_api_key and not settings.api.openai_api_key:
        issues.append("At least one LLM API key is required (DeepSeek or OpenAI)")
    
    if not settings.api.pubmed_email:
        issues.append("PubMed email is required")
    
    # Check paths
    if not settings.paths.project_root:
        issues.append("Project root is required")
    
    return issues