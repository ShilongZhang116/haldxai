# -*- coding: utf-8 -*-
"""
HALDxAI: Healthy Aging and Longevity Discovery AI Platform

A comprehensive platform for extracting, analyzing, and visualizing
aging-related knowledge from scientific literature.
"""

__version__ = "0.1.0"
__author__ = "HALDxAI Development Team"
__email__ = "haldxai@example.com"

# Core imports
from .core import HALDxAI
from .config import Settings, load_config

# Main functionality
from .ner import LLMNER, SpacyNER
from .database import PostgreSQLManager, Neo4jManager
from .modeling import AgingClassifier, RelationExtractor
from .scoring import AgingScorer, BridgeCandidateScorer
from .selection import SeedSelector, EntitySelector
from .visualization import NetworkVisualizer, PlotGenerator

# Utilities
from .utils import setup_logging, get_project_root

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    
    # Core classes
    "HALDxAI",
    "Settings",
    "load_config",
    
    # Main functionality
    "LLMNER",
    "SpacyNER", 
    "PostgreSQLManager",
    "Neo4jManager",
    "AgingClassifier",
    "RelationExtractor",
    "AgingScorer",
    "BridgeCandidateScorer",
    "SeedSelector",
    "EntitySelector",
    "NetworkVisualizer",
    "PlotGenerator",
    
    # Utilities
    "setup_logging",
    "get_project_root",
]

# Package metadata
PACKAGE_INFO = {
    "name": "HALDxAI",
    "description": "Healthy Aging and Longevity Discovery AI Platform",
    "url": "https://github.com/your-org/HALDxAI-Repository",
    "license": "MIT",
    "python_requires": ">=3.8",
}

def get_version():
    """Get package version."""
    return __version__

def get_info():
    """Get package information."""
    return PACKAGE_INFO.copy()