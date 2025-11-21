# -*- coding: utf-8 -*-
"""
Entity management for HALDxAI platform.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Set
from datetime import datetime
import uuid

from .types import EntityID, EntityType, ConfidenceScore, Evidence, ProcessingResult


@dataclass
class Entity:
    """Represents an entity in the HALD system."""
    id: EntityID
    name: str
    entity_type: EntityType
    description: Optional[str] = None
    synonyms: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    confidence: ConfidenceScore = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def add_evidence(self, evidence: Evidence) -> None:
        """Add evidence supporting this entity."""
        self.evidence.append(evidence)
        self.updated_at = datetime.now()
    
    def add_synonym(self, synonym: str) -> None:
        """Add a synonym for this entity."""
        if synonym not in self.synonyms:
            self.synonyms.append(synonym)
            self.updated_at = datetime.now()
    
    def get_all_names(self) -> Set[str]:
        """Get all names for this entity (primary + synonyms)."""
        return {self.name} | set(self.synonyms)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert entity to dictionary representation."""
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type.value,
            "description": self.description,
            "synonyms": self.synonyms,
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
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class EntityManager:
    """Manages entities in the HALD system."""
    
    def __init__(self):
        self._entities: Dict[EntityID, Entity] = {}
        self._name_index: Dict[str, EntityID] = {}
        self._type_index: Dict[EntityType, Set[EntityID]] = {}
    
    def add_entity(self, entity: Entity) -> ProcessingResult:
        """Add an entity to the manager."""
        try:
            # Check for duplicates by name
            for name in entity.get_all_names():
                if name in self._name_index:
                    existing_id = self._name_index[name]
                    existing_entity = self._entities[existing_id]
                    # Merge with existing entity
                    return self._merge_entities(existing_entity, entity)
            
            # Add new entity
            self._entities[entity.id] = entity
            
            # Update indexes
            for name in entity.get_all_names():
                self._name_index[name] = entity.id
            
            if entity.entity_type not in self._type_index:
                self._type_index[entity.entity_type] = set()
            self._type_index[entity.entity_type].add(entity.id)
            
            return ProcessingResult(
                success=True,
                message=f"Entity {entity.id} added successfully",
                data=entity
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"Failed to add entity: {str(e)}",
                errors=[str(e)]
            )
    
    def get_entity(self, entity_id: EntityID) -> Optional[Entity]:
        """Get an entity by ID."""
        return self._entities.get(entity_id)
    
    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        """Get an entity by name (primary or synonym)."""
        entity_id = self._name_index.get(name)
        if entity_id:
            return self._entities.get(entity_id)
        return None
    
    def get_entities_by_type(self, entity_type: EntityType) -> List[Entity]:
        """Get all entities of a specific type."""
        entity_ids = self._type_index.get(entity_type, set())
        return [self._entities[eid] for eid in entity_ids]
    
    def search_entities(self, query: str, entity_types: Optional[List[EntityType]] = None) -> List[Entity]:
        """Search entities by name or description."""
        query_lower = query.lower()
        results = []
        
        for entity in self._entities.values():
            # Filter by type if specified
            if entity_types and entity.entity_type not in entity_types:
                continue
            
            # Search in name, synonyms, and description
            if (query_lower in entity.name.lower() or
                any(query_lower in syn.lower() for syn in entity.synonyms) or
                (entity.description and query_lower in entity.description.lower())):
                results.append(entity)
        
        return results
    
    def update_entity(self, entity_id: EntityID, updates: Dict[str, Any]) -> ProcessingResult:
        """Update an entity with new data."""
        if entity_id not in self._entities:
            return ProcessingResult(
                success=False,
                message=f"Entity {entity_id} not found"
            )
        
        try:
            entity = self._entities[entity_id]
            
            # Update fields
            for key, value in updates.items():
                if hasattr(entity, key):
                    setattr(entity, key, value)
            
            entity.updated_at = datetime.now()
            
            # Rebuild indexes if name changed
            if "name" in updates or "synonyms" in updates:
                self._rebuild_name_index()
            
            return ProcessingResult(
                success=True,
                message=f"Entity {entity_id} updated successfully",
                data=entity
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"Failed to update entity: {str(e)}",
                errors=[str(e)]
            )
    
    def delete_entity(self, entity_id: EntityID) -> ProcessingResult:
        """Delete an entity from the manager."""
        if entity_id not in self._entities:
            return ProcessingResult(
                success=False,
                message=f"Entity {entity_id} not found"
            )
        
        try:
            entity = self._entities[entity_id]
            
            # Remove from indexes
            for name in entity.get_all_names():
                self._name_index.pop(name, None)
            
            self._type_index[entity.entity_type].discard(entity_id)
            
            # Remove entity
            del self._entities[entity_id]
            
            return ProcessingResult(
                success=True,
                message=f"Entity {entity_id} deleted successfully"
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"Failed to delete entity: {str(e)}",
                errors=[str(e)]
            )
    
    def get_all_entities(self) -> List[Entity]:
        """Get all entities."""
        return list(self._entities.values())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about entities."""
        type_counts = {}
        for entity_type, entity_ids in self._type_index.items():
            type_counts[entity_type.value] = len(entity_ids)
        
        return {
            "total_entities": len(self._entities),
            "type_counts": type_counts,
            "total_synonyms": sum(len(e.synonyms) for e in self._entities.values()),
            "average_confidence": sum(e.confidence for e in self._entities.values()) / len(self._entities) if self._entities else 0
        }
    
    def _merge_entities(self, existing: Entity, new: Entity) -> ProcessingResult:
        """Merge two entities."""
        try:
            # Merge evidence
            existing.evidence.extend(new.evidence)
            
            # Merge synonyms
            for synonym in new.synonyms:
                existing.add_synonym(synonym)
            
            # Update confidence if higher
            if new.confidence > existing.confidence:
                existing.confidence = new.confidence
            
            # Update description if new one is better
            if new.description and (not existing.description or len(new.description) > len(existing.description)):
                existing.description = new.description
            
            # Merge metadata
            existing.metadata.update(new.metadata)
            existing.updated_at = datetime.now()
            
            return ProcessingResult(
                success=True,
                message=f"Entity {existing.id} merged successfully",
                data=existing
            )
        except Exception as e:
            return ProcessingResult(
                success=False,
                message=f"Failed to merge entities: {str(e)}",
                errors=[str(e)]
            )
    
    def _rebuild_name_index(self) -> None:
        """Rebuild the name index."""
        self._name_index.clear()
        for entity_id, entity in self._entities.items():
            for name in entity.get_all_names():
                self._name_index[name] = entity_id
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EntityManager':
        """Create EntityManager from dictionary data."""
        manager = cls()
        
        for entity_data in data.get("entities", []):
            entity = Entity(
                id=entity_data["id"],
                name=entity_data["name"],
                entity_type=EntityType(entity_data["entity_type"]),
                description=entity_data.get("description"),
                synonyms=entity_data.get("synonyms", []),
                confidence=entity_data.get("confidence", 0.0),
                metadata=entity_data.get("metadata", {})
            )
            manager.add_entity(entity)
        
        return manager
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert EntityManager to dictionary representation."""
        return {
            "entities": [entity.to_dict() for entity in self._entities.values()],
            "statistics": self.get_statistics()
        }