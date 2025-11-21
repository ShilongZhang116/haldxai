# -*- coding: utf-8 -*-
"""
Utility functions for HALDxAI platform.
"""

from .io import load_json, save_json, load_csv, save_csv
from .logging import setup_logging, get_logger
from .helpers import (
    get_project_root,
    ensure_directory,
    clean_text,
    normalize_text,
    validate_email,
    validate_api_key,
    retry_with_backoff,
    chunk_list,
    flatten_dict,
    merge_dicts,
)

__all__ = [
    # IO utilities
    "load_json",
    "save_json", 
    "load_csv",
    "save_csv",
    
    # Logging utilities
    "setup_logging",
    "get_logger",
    
    # Helper utilities
    "get_project_root",
    "ensure_directory",
    "clean_text",
    "normalize_text",
    "validate_email",
    "validate_api_key",
    "retry_with_backoff",
    "chunk_list",
    "flatten_dict",
    "merge_dicts",
]