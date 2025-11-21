# -*- coding: utf-8 -*-
"""
Type definitions and enums for HALDxAI.
"""

from enum import Enum
from typing import List, Dict, Optional, Union, Any
from dataclasses import dataclass


class EntityType(Enum):
    """Entity types in the HALD system."""
    BMC = "BMC"  # Biological Molecular Components
    EGR = "EGR"  # Epigenetic Regulators
    ASPKM = "ASPKM"  # Aging Signaling Pathways and Kinase Modulators
    CRD = "CRD"  # Cellular Repair and Defense
    APP = "APP"  # Aging Protective Processes
    SCN = "SCN"  # Stem Cell Niches
    AAI = "AAI"  # Anti-Aging Interventions
    CRBC = "CRBC"  # Cellular Regeneration and Brain Cells
    NM = "NM"  # Neurotransmission and Metabolism
    EF = "EF"  # Environmental Factors


class RelationType(Enum):
    """Relation types in the HALD system."""
    REGULATES = "regulates"
    INTERACTS_WITH = "interacts_with"
    INHIBITS = "inhibits"
    ACTIVATES = "activates"
    BINDS_TO = "binds_to"
    PART_OF = "part_of"
    ASSOCIATED_WITH = "associated_with"
    TREATS = "treats"
    PREVENTS = "prevents"
    CAUSES = "causes"


@dataclass
class HALDClasses:
    """HALD class definitions and properties."""
    CLASSES: List[str] = [
        "BMC", "EGR", "ASPKM", "CRD", "APP",
        "SCN", "AAI", "CRBC", "NM", "EF",
    ]
    
    COLOR_MAP: Dict[str, str] = {
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
    }
    
    DESCRIPTIONS: Dict[str, str] = {
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
    }


@dataclass
class Evidence:
    """Evidence supporting an entity or relation."""
    pmid: str
    text: str
    confidence: float
    source: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class DataSource:
    """Data source information."""
    name: str
    version: str
    url: Optional[str] = None
    description: Optional[str] = None
    last_updated: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of a processing operation."""
    success: bool
    message: str
    data: Optional[Any] = None
    errors: Optional[List[str]] = None
    warnings: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class ProcessingStatus(Enum):
    """Status of processing operations."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TaskConfig:
    """Configuration for processing tasks."""
    name: str
    parameters: Dict[str, Any]
    dependencies: Optional[List[str]] = None
    timeout: Optional[int] = None
    retry_count: int = 3
    priority: int = 0


# Type aliases for better readability
EntityID = str
RelationID = str
PMID = str
ConfidenceScore = float
Timestamp = str
JSONDict = Dict[str, Any]