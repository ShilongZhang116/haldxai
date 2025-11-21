# -*- coding: utf-8 -*-
"""
Relation management for HALDxAI platform.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set
from datetime import datetime
import uuid

from .types import RelationID, RelationType, ConfidenceScore, Evidence, ProcessingResult


@dataclass
class Relation:
    """Represents a relation between entities in HALD system."""
    id: RelationID
    source_id: str
    target_id: str
    relation_type: RelationType
    description: Optional[str] = None
    evidence: List[Evidence] = field(default_factory=list)
    confidence: ConfidenceScore = 0.0
    direction: str = "directed"  # "directed" or "undirected"
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_evidence(self, evidence: Evidence) -> None:
        """Add evidence supporting this relation."""
        self.evidence.append(evidence)
        self.updated_at = datetime.now()
    
    def get_entities(self) -> tuple[str, str]:
        """Get the source and target entity IDs."""
        return self.source_id, self.target_id
    
    def is_directed(self) -> bool:
        """Check if relation is directed."""
        return self.direction == "directed"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert relation to dictionary representation."""
        return {
            "id": self.id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "description": self.description,
            "evidence": [
                {
                    "pmid": e.pmid,
                    "text": e.text,
                    "confidence": e.confidence,
                    "source": e.source,
                    "metadata": e.metadata
                } for e in self.evidence
            ],
            "confidence": self.confidence,
            "direction": self.direction,
            "weight": self.weight,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class RelationManager:
    """Manages relations between entities in HALD system."""
    
    def __init__(self):
        self._relations: Dict[RelationID, Relation] = {}
        self._entity_relations: Dict[str, Set[RelationID]] = {}
        self._type_index: Dict[RelationType, Set[RelationID]] = {}
    
    def add_relation(self, relation: Relation) -> ProcessingResult:
        """Add a relation to the manager."""
        try:
            # Check for duplicates
            existing_id = self._find_duplicate_relation(relation)
            if existing_id:
                existing_relation = self._relations[existing_id]
                # Merge with existing relation
                return self._merge_relations(existing_relation, relation)
            
            # Add new relation
            self._relations[relation.id] = relation
            
            # Update indexes
            for entity_id in [relation.source_id, relation.target_id]:
                if entity_id not in self._entity_relations:
                    self._entity_relations[entity_id] = set()
                self._entity_relations[entity_id].add(relation.id)
            
            if relation.relation_type not in self._type_index:
                self._type_index[relation.relation_type] = set()
            self._type_index[relation.relation_type].add(relation.id)
            
            return ProcessingResult(
                success=True,
                message=f"Relation {relation.id} added successfully",
                data=relation
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"Failed to add relation: {str(e)}",
                errors=[str(e)]
            )
    
    def get_relation(self, relation_id: RelationID) -> Optional[Relation]:
        """Get a relation by ID."""
        return self._relations.get(relation_id)
    
    def get_relations_between_entities(self, source_id: str, target_id: str) -> List[Relation]:
        """Get all relations between two entities."""
        source_relations = self._entity_relations.get(source_id, set())
        target_relations = self._entity_relations.get(target_id, set())
        common_relation_ids = source_relations & target_relations
        
        return [self._relations[rel_id] for rel_id in common_relation_ids]
    
    def get_entity_relations(self, entity_id: str, relation_type: Optional[RelationType] = None) -> List[Relation]:
        """Get all relations for an entity, optionally filtered by type."""
        relation_ids = self._entity_relations.get(entity_id, set())
        relations = [self._relations[rel_id] for rel_id in relation_ids]
        
        if relation_type:
            relations = [rel for rel in relations if rel.relation_type == relation_type]
        
        return relations
    
    def get_relations_by_type(self, relation_type: RelationType) -> List[Relation]:
        """Get all relations of a specific type."""
        relation_ids = self._type_index.get(relation_type, set())
        return [self._relations[rel_id] for rel_id in relation_ids]
    
    def search_relations(self, query: str, entity_ids: Optional[List[str]] = None) -> List[Relation]:
        """Search relations by description or metadata."""
        query_lower = query.lower()
        results = []
        
        for relation in self._relations.values():
            # Filter by entities if specified
            if entity_ids:
                if relation.source_id not in entity_ids and relation.target_id not in entity_ids:
                    continue
            
            # Search in description and metadata
            if (relation.description and query_lower in relation.description.lower() or
                any(query_lower in str(v).lower() for v in relation.metadata.values())):
                results.append(relation)
        
        return results
    
    def update_relation(self, relation_id: RelationID, updates: Dict[str, Any]) -> ProcessingResult:
        """Update a relation with new data."""
        if relation_id not in self._relations:
            return ProcessingResult(
                success=False,
                message=f"Relation {relation_id} not found"
            )
        
        try:
            relation = self._relations[relation_id]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(relation, key):
                    setattr(relation, key, value)
            
            relation.updated_at = datetime.now()
            
            # Rebuild type index if relation type changed
            if "relation_type" in updates:
                self._rebuild_type_index()
            
            return ProcessingResult(
                success=True,
                message=f"Relation {relation_id} updated successfully",
                data=relation
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"Failed to update relation: {str(e)}",
                errors=[str(e)]
            )
    
    def delete_relation(self, relation_id: RelationID) -> ProcessingResult:
        """Delete a relation from the manager."""
        if relation_id not in self._relations:
            return ProcessingResult(
                success=False,
                message=f"Relation {relation_id} not found"
            )
        
        try:
            relation = self._relations[relation_id]
            
            # Remove from entity index
            for entity_id in [relation.source_id, relation.target_id]:
                self._entity_relations[entity_id].discard(relation_id)
            
            # Remove from type index
            self._type_index[relation.relation_type].discard(relation_id)
            
            # Remove relation
            del self._relations[relation_id]
            
            return ProcessingResult(
                success=True,
                message=f"Relation {relation_id} deleted successfully"
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"Failed to delete relation: {str(e)}",
                errors=[str(e)]
            )
    
    def get_all_relations(self) -> List[Relation]:
        """Get all relations."""
        return list(self._relations.values())
    
    def get_network_data(self) -> Dict[str, Any]:
        """Get network data for visualization."""
        nodes = set()
        edges = []
        
        for relation in self._relations.values():
            nodes.add(relation.source_id)
            nodes.add(relation.target_id)
            edges.append({
                "source": relation.source_id,
                "target": relation.target_id,
                "type": relation.relation_type.value,
                "weight": relation.weight,
                "confidence": relation.confidence,
                "directed": relation.is_directed()
            })
        
        return {
            "nodes": list(nodes),
            "edges": edges,
            "statistics": self.get_statistics()
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about relations."""
        type_counts = {}
        for relation_type, relation_ids in self._type_index.items():
            type_counts[relation_type.value] = len(relation_ids)
        
        directed_count = sum(1 for rel in self._relations.values() if rel.is_directed())
        
        return {
            "total_relations": len(self._relations),
            "type_counts": type_counts,
            "directed_relations": directed_count,
            "undirected_relations": len(self._relations) - directed_count,
            "average_confidence": sum(rel.confidence for rel in self._relations.values()) / len(self._relations) if self._relations else 0,
            "connected_entities": len(self._entity_relations)
        }
    
    def _find_duplicate_relation(self, relation: Relation) -> Optional[RelationID]:
        """Find if a duplicate relation exists."""
        for rel_id, existing_rel in self._relations.items():
            if (existing_rel.source_id == relation.source_id and
                existing_rel.target_id == relation.target_id and
                existing_rel.relation_type == relation.relation_type):
                return rel_id
        return None
    
    def _merge_relations(self, existing: Relation, new: Relation) -> ProcessingResult:
        """Merge two relations."""
        try:
            # Merge evidence
            existing.evidence.extend(new.evidence)
            
            # Update confidence if higher
            if new.confidence > existing.confidence:
                existing.confidence = new.confidence
            
            # Update weight if higher
            if new.weight > existing.weight:
                existing.weight = new.weight
            
            # Update description if new one is better
            if new.description and (not existing.description or len(new.description) > len(existing.description)):
                existing.description = new.description
            
            # Merge metadata
            existing.metadata.update(new.metadata)
            existing.updated_at = datetime.now()
            
            return ProcessingResult(
                success=True,
                message=f"Relation {existing.id} merged successfully",
                data=existing
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"Failed to merge relations: {str(e)}",
                errors=[str(e)]
            )
    
    def _rebuild_type_index(self) -> None:
        """Rebuild type index."""
        self._type_index.clear()
        for relation_id, relation in self._relations.items():
            if relation.relation_type not in self._type_index:
                self._type_index[relation.relation_type] = set()
            self._type_index[relation.relation_type].add(relation_id)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RelationManager':
        """Create RelationManager from dictionary data."""
        manager = cls()
        
        for relation_data in data.get("relations", []):
            relation = Relation(
                id=relation_data["id"],
                source_id=relation_data["source_id"],
                target_id=relation_data["target_id"],
                relation_type=RelationType(relation_data["relation_type"]),
                description=relation_data.get("description"),
                confidence=relation_data.get("confidence", 0.0),
                direction=relation_data.get("direction", "directed"),
                weight=relation_data.get("weight", 1.0),
                metadata=relation_data.get("metadata", {})
            )
            manager.add_relation(relation)
        
        return manager
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert RelationManager to dictionary representation."""
        return {
            "relations": [relation.to_dict() for relation in self._relations.values()],
            "statistics": self.get_statistics()
        }