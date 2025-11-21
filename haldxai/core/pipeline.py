# -*- coding: utf-8 -*-
"""
Main HALDxAI pipeline for orchestrating the entire workflow.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import logging
from dataclasses import dataclass

from .entities import EntityManager
from .relations import RelationManager
from .types import ProcessingResult, ProcessingStatus
from ..config import Settings, load_config
from ..utils import setup_logging


@dataclass
class PipelineConfig:
    """Configuration for HALDxAI pipeline."""
    config_path: Optional[Union[str, Path]] = None
    data_dir: Optional[Union[str, Path]] = None
    output_dir: Optional[Union[str, Path]] = None
    log_level: str = "INFO"
    enable_caching: bool = True
    parallel_processing: bool = True
    max_workers: int = 4


class HALDxAI:
    """Main HALDxAI platform class."""
    
    def __init__(self, config: Optional[Union[str, Path, PipelineConfig, Settings]] = None):
        """Initialize HALDxAI platform."""
        # Load configuration
        if isinstance(config, PipelineConfig):
            self.config = self._load_config_from_pipeline_config(config)
        elif isinstance(config, Settings):
            self.config = config
        else:
            self.config = load_config(config)
        
        # Setup logging
        setup_logging(self.config.log_level)
        self.logger = logging.getLogger(__name__)
        
        # Initialize managers
        self.entity_manager = EntityManager()
        self.relation_manager = RelationManager()
        
        # Initialize components
        self._ner_components = {}
        self._modeling_components = {}
        self._scoring_components = {}
        self._visualization_components = {}
        
        self.logger.info("HALDxAI platform initialized")
    
    def run_ner_pipeline(self, 
                        input_file: Union[str, Path], 
                        output_dir: Union[str, Path],
                        ner_method: str = "llm",
                        **kwargs) -> ProcessingResult:
        """Run Named Entity Recognition pipeline."""
        try:
            self.logger.info(f"Starting NER pipeline with method: {ner_method}")
            
            # Import NER components
            if ner_method == "llm":
                from ..ner import LLMNER
                ner_component = LLMNER(self.config)
            elif ner_method == "spacy":
                from ..ner import SpacyNER
                ner_component = SpacyNER(self.config)
            else:
                raise ValueError(f"Unknown NER method: {ner_method}")
            
            # Run NER
            results = ner_component.process_file(input_file, output_dir, **kwargs)
            
            # Update managers with results
            if results.success and results.data:
                entities = results.data.get("entities", [])
                for entity_data in entities:
                    self.entity_manager.add_entity(entity_data)
            
            self.logger.info(f"NER pipeline completed: {results.message}")
            return results
            
        except Exception as e:
            self.logger.error(f"NER pipeline failed: {str(e)}")
            return ProcessingResult(
                success=False,
                message=f"NER pipeline failed: {str(e)}",
                errors=[str(e)]
            )
    
    def run_relation_extraction(self,
                             entities_file: Union[str, Path],
                             output_dir: Union[str, Path],
                             method: str = "llm",
                             **kwargs) -> ProcessingResult:
        """Run relation extraction pipeline."""
        try:
            self.logger.info(f"Starting relation extraction with method: {method}")
            
            # Import relation extraction components
            if method == "llm":
                from ..modeling import RelationExtractor
                extractor = RelationExtractor(self.config)
            else:
                raise ValueError(f"Unknown relation extraction method: {method}")
            
            # Run extraction
            results = extractor.extract_relations(entities_file, output_dir, **kwargs)
            
            # Update managers with results
            if results.success and results.data:
                relations = results.data.get("relations", [])
                for relation_data in relations:
                    self.relation_manager.add_relation(relation_data)
            
            self.logger.info(f"Relation extraction completed: {results.message}")
            return results
            
        except Exception as e:
            self.logger.error(f"Relation extraction failed: {str(e)}")
            return ProcessingResult(
                success=False,
                message=f"Relation extraction failed: {str(e)}",
                errors=[str(e)]
            )
    
    def run_scoring_pipeline(self,
                           output_dir: Union[str, Path],
                           scoring_types: List[str] = None,
                           **kwargs) -> ProcessingResult:
        """Run scoring pipeline for entities and relations."""
        try:
            self.logger.info("Starting scoring pipeline")
            
            if scoring_types is None:
                scoring_types = ["aging_relevance", "bridge_candidates"]
            
            results = {"entities": {}, "relations": {}}
            
            # Entity scoring
            if "aging_relevance" in scoring_types:
                from ..scoring import AgingScorer
                scorer = AgingScorer(self.config)
                score_results = scorer.score_entities(self.entity_manager.get_all_entities())
                results["entities"]["aging_relevance"] = score_results
            
            # Bridge candidate scoring
            if "bridge_candidates" in scoring_types:
                from ..scoring import BridgeCandidateScorer
                scorer = BridgeCandidateScorer(self.config)
                score_results = scorer.score_bridge_candidates(
                    self.entity_manager.get_all_entities(),
                    self.relation_manager.get_all_relations()
                )
                results["relations"]["bridge_candidates"] = score_results
            
            # Save results
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(output_path / "scoring_results.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
            
            self.logger.info("Scoring pipeline completed successfully")
            return ProcessingResult(
                success=True,
                message="Scoring pipeline completed successfully",
                data=results
            )
            
        except Exception as e:
            self.logger.error(f"Scoring pipeline failed: {str(e)}")
            return ProcessingResult(
                success=False,
                message=f"Scoring pipeline failed: {str(e)}",
                errors=[str(e)]
            )
    
    def build_knowledge_graph(self, 
                           entities_file: Optional[Union[str, Path]] = None,
                           relations_file: Optional[Union[str, Path]] = None,
                           output_dir: Optional[Union[str, Path]] = None) -> ProcessingResult:
        """Build knowledge graph from entities and relations."""
        try:
            self.logger.info("Building knowledge graph")
            
            # Load data if files provided
            if entities_file:
                from ..data import DataLoader
                loader = DataLoader(self.config)
                entity_results = loader.load_entities(entities_file)
                if entity_results.success:
                    for entity in entity_results.data:
                        self.entity_manager.add_entity(entity)
            
            if relations_file:
                from ..data import DataLoader
                loader = DataLoader(self.config)
                relation_results = loader.load_relations(relations_file)
                if relation_results.success:
                    for relation in relation_results.data:
                        self.relation_manager.add_relation(relation)
            
            # Build graph data
            graph_data = self.relation_manager.get_network_data()
            
            # Add entity information
            entity_info = {}
            for entity in self.entity_manager.get_all_entities():
                entity_info[entity.id] = {
                    "name": entity.name,
                    "type": entity.entity_type.value,
                    "confidence": entity.confidence,
                    "description": entity.description
                }
            
            graph_data["entity_info"] = entity_info
            
            # Save if output directory provided
            if output_dir:
                output_path = Path(output_dir)
                output_path.mkdir(parents=True, exist_ok=True)
                
                import json
                with open(output_path / "knowledge_graph.json", "w") as f:
                    json.dump(graph_data, f, indent=2, default=str)
            
            self.logger.info("Knowledge graph built successfully")
            return ProcessingResult(
                success=True,
                message="Knowledge graph built successfully",
                data=graph_data
            )
            
        except Exception as e:
            self.logger.error(f"Knowledge graph construction failed: {str(e)}")
            return ProcessingResult(
                success=False,
                message=f"Knowledge graph construction failed: {str(e)}",
                errors=[str(e)]
            )
    
    def generate_report(self, 
                      output_path: Union[str, Path],
                      report_type: str = "html",
                      include_visualizations: bool = True) -> ProcessingResult:
        """Generate comprehensive analysis report."""
        try:
            self.logger.info(f"Generating {report_type} report")
            
            # Collect statistics
            entity_stats = self.entity_manager.get_statistics()
            relation_stats = self.relation_manager.get_statistics()
            
            report_data = {
                "generated_at": str(Path().resolve()),
                "entity_statistics": entity_stats,
                "relation_statistics": relation_stats,
                "configuration": self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
            }
            
            # Generate visualizations if requested
            if include_visualizations:
                from ..visualization import NetworkVisualizer
                visualizer = NetworkVisualizer(self.config)
                
                # Generate network visualization
                network_data = self.relation_manager.get_network_data()
                viz_results = visualizer.create_network_plot(network_data, output_path.parent)
                if viz_results.success:
                    report_data["visualizations"] = viz_results.data
            
            # Generate report
            if report_type == "html":
                report_content = self._generate_html_report(report_data)
            elif report_type == "json":
                import json
                report_content = json.dumps(report_data, indent=2, default=str)
            else:
                raise ValueError(f"Unknown report type: {report_type}")
            
            # Save report
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(report_content, encoding="utf-8")
            
            self.logger.info(f"Report generated: {output_path}")
            return ProcessingResult(
                success=True,
                message=f"Report generated successfully: {output_path}",
                data=report_data
            )
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            return ProcessingResult(
                success=False,
                message=f"Report generation failed: {str(e)}",
                errors=[str(e)]
            )
    
    def save_data(self, output_dir: Union[str, Path]) -> ProcessingResult:
        """Save all data to files."""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save entities
            entity_data = self.entity_manager.to_dict()
            with open(output_path / "entities.json", "w") as f:
                import json
                json.dump(entity_data, f, indent=2, default=str)
            
            # Save relations
            relation_data = self.relation_manager.to_dict()
            with open(output_path / "relations.json", "w") as f:
                json.dump(relation_data, f, indent=2, default=str)
            
            self.logger.info(f"Data saved to: {output_path}")
            return ProcessingResult(
                success=True,
                message=f"Data saved successfully to: {output_path}",
                data={"entities": entity_data, "relations": relation_data}
            )
            
        except Exception as e:
            self.logger.error(f"Data saving failed: {str(e)}")
            return ProcessingResult(
                success=False,
                message=f"Data saving failed: {str(e)}",
                errors=[str(e)]
            )
    
    def load_data(self, input_dir: Union[str, Path]) -> ProcessingResult:
        """Load data from files."""
        try:
            input_path = Path(input_dir)
            
            # Load entities
            entities_file = input_path / "entities.json"
            if entities_file.exists():
                with open(entities_file, "r") as f:
                    import json
                    entity_data = json.load(f)
                self.entity_manager = EntityManager.from_dict(entity_data)
            
            # Load relations
            relations_file = input_path / "relations.json"
            if relations_file.exists():
                with open(relations_file, "r") as f:
                    import json
                    relation_data = json.load(f)
                self.relation_manager = RelationManager.from_dict(relation_data)
            
            self.logger.info(f"Data loaded from: {input_path}")
            return ProcessingResult(
                success=True,
                message=f"Data loaded successfully from: {input_path}"
            )
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            return ProcessingResult(
                success=False,
                message=f"Data loading failed: {str(e)}",
                errors=[str(e)]
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "entities": self.entity_manager.get_statistics(),
            "relations": self.relation_manager.get_statistics(),
            "configuration": self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
        }
    
    def _load_config_from_pipeline_config(self, pipeline_config: PipelineConfig) -> Settings:
        """Load Settings from PipelineConfig."""
        if pipeline_config.config_path:
            config = load_config(pipeline_config.config_path)
        else:
            config = Settings()
        
        # Override with pipeline config
        if pipeline_config.data_dir:
            config.data_dir = str(pipeline_config.data_dir)
        if pipeline_config.output_dir:
            config.output_dir = str(pipeline_config.output_dir)
        if pipeline_config.log_level:
            config.log_level = pipeline_config.log_level
        if pipeline_config.enable_caching is not None:
            config.enable_caching = pipeline_config.enable_caching
        if pipeline_config.parallel_processing is not None:
            config.parallel_processing = pipeline_config.parallel_processing
        if pipeline_config.max_workers:
            config.max_workers = pipeline_config.max_workers
        
        return config
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML report."""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>HALDxAI Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
                .section { margin: 20px 0; }
                .stats-table { border-collapse: collapse; width: 100%; }
                .stats-table th, .stats-table td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                .stats-table th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>HALDxAI Analysis Report</h1>
                <p>Generated at: {generated_at}</p>
            </div>
            
            <div class="section">
                <h2>Entity Statistics</h2>
                <table class="stats-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Entities</td><td>{total_entities}</td></tr>
                    <tr><td>Total Synonyms</td><td>{total_synonyms}</td></tr>
                    <tr><td>Average Confidence</td><td>{average_confidence:.3f}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Relation Statistics</h2>
                <table class="stats-table">
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Total Relations</td><td>{total_relations}</td></tr>
                    <tr><td>Directed Relations</td><td>{directed_relations}</td></tr>
                    <tr><td>Connected Entities</td><td>{connected_entities}</td></tr>
                </table>
            </div>
        </body>
        </html>
        """
        
        # Extract statistics
        entity_stats = report_data.get("entity_statistics", {})
        relation_stats = report_data.get("relation_statistics", {})
        
        return html_template.format(
            generated_at=report_data.get("generated_at", "Unknown"),
            total_entities=entity_stats.get("total_entities", 0),
            total_synonyms=entity_stats.get("total_synonyms", 0),
            average_confidence=entity_stats.get("average_confidence", 0),
            total_relations=relation_stats.get("total_relations", 0),
            directed_relations=relation_stats.get("directed_relations", 0),
            connected_entities=relation_stats.get("connected_entities", 0)
        )