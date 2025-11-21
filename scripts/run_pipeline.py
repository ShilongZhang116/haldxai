#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HALDxAI pipeline execution script.
"""

import sys
import argparse
from pathlib import Path
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from haldxai.core.pipeline import HALDxAI, PipelineConfig
from haldxai.config import load_config
from haldxai.utils import setup_logging, get_logger


def run_complete_pipeline(config, input_file, output_dir, options):
    """Run complete HALDxAI pipeline."""
    logger = get_logger(__name__)
    
    try:
        # Initialize HALDxAI
        hald = HALDxAI(config)
        
        logger.info("Starting complete HALDxAI pipeline...")
        start_time = time.time()
        
        # Step 1: Named Entity Recognition
        logger.info("Step 1: Running Named Entity Recognition...")
        ner_method = options.get('ner_method', 'llm')
        ner_results = hald.run_ner_pipeline(input_file, output_dir, ner_method)
        
        if not ner_results.success:
            logger.error(f"NER pipeline failed: {ner_results.message}")
            return False
        
        logger.info(f"NER completed: {ner_results.message}")
        
        # Step 2: Relation Extraction
        if options.get('skip_relations', False):
            logger.info("Skipping relation extraction...")
        else:
            logger.info("Step 2: Extracting relations...")
            entities_file = Path(output_dir) / "entities.json"
            rel_results = hald.run_relation_extraction(entities_file, output_dir)
            
            if not rel_results.success:
                logger.error(f"Relation extraction failed: {rel_results.message}")
                return False
            
            logger.info(f"Relation extraction completed: {rel_results.message}")
        
        # Step 3: Scoring
        if options.get('skip_scoring', False):
            logger.info("Skipping scoring...")
        else:
            logger.info("Step 3: Running scoring...")
            scoring_types = options.get('scoring_types', ['aging_relevance', 'bridge_candidates'])
            score_results = hald.run_scoring_pipeline(output_dir, scoring_types)
            
            if not score_results.success:
                logger.error(f"Scoring failed: {score_results.message}")
                return False
            
            logger.info(f"Scoring completed: {score_results.message}")
        
        # Step 4: Knowledge Graph Construction
        logger.info("Step 4: Building knowledge graph...")
        kg_results = hald.build_knowledge_graph(output_dir=output_dir)
        
        if not kg_results.success:
            logger.error(f"Knowledge graph construction failed: {kg_results.message}")
            return False
        
        logger.info(f"Knowledge graph built: {kg_results.message}")
        
        # Step 5: Report Generation
        if options.get('skip_report', False):
            logger.info("Skipping report generation...")
        else:
            logger.info("Step 5: Generating report...")
            report_path = Path(output_dir) / "pipeline_report.html"
            report_results = hald.generate_report(report_path)
            
            if not report_results.success:
                logger.error(f"Report generation failed: {report_results.message}")
                return False
            
            logger.info(f"Report generated: {report_results.message}")
        
        # Step 6: Save Data
        if options.get('save_data', True):
            logger.info("Step 6: Saving processed data...")
            save_results = hald.save_data(output_dir)
            
            if not save_results.success:
                logger.error(f"Data saving failed: {save_results.message}")
                return False
            
            logger.info(f"Data saved: {save_results.message}")
        
        end_time = time.time()
        duration = end_time - start_time
        
        logger.info(f"Pipeline completed successfully in {duration:.2f} seconds")
        logger.info(f"Results saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return False


def run_ner_only(config, input_file, output_dir, options):
    """Run only NER pipeline."""
    logger = get_logger(__name__)
    
    try:
        hald = HALDxAI(config)
        ner_method = options.get('ner_method', 'llm')
        
        logger.info("Running NER pipeline only...")
        results = hald.run_ner_pipeline(input_file, output_dir, ner_method)
        
        if results.success:
            logger.info(f"NER completed: {results.message}")
            return True
        else:
            logger.error(f"NER failed: {results.message}")
            return False
            
    except Exception as e:
        logger.error(f"NER execution failed: {e}")
        return False


def run_analysis_only(config, data_dir, output_dir, options):
    """Run analysis only on existing data."""
    logger = get_logger(__name__)
    
    try:
        hald = HALDxAI(config)
        
        logger.info("Loading existing data...")
        load_results = hald.load_data(data_dir)
        
        if not load_results.success:
            logger.error(f"Data loading failed: {load_results.message}")
            return False
        
        logger.info("Generating analysis report...")
        report_path = Path(output_dir) / "analysis_report.html"
        report_results = hald.generate_report(report_path)
        
        if report_results.success:
            logger.info(f"Analysis completed: {report_results.message}")
            return True
        else:
            logger.error(f"Analysis failed: {report_results.message}")
            return False
            
    except Exception as e:
        logger.error(f"Analysis execution failed: {e}")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run HALDxAI pipeline")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--input", "-i", required=True, help="Input file or directory")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    # Pipeline options
    parser.add_argument("--mode", choices=['complete', 'ner', 'analysis'], 
                       default='complete', help="Pipeline mode to run")
    parser.add_argument("--ner-method", choices=['llm', 'spacy'], 
                       default='llm', help="NER method to use")
    parser.add_argument("--skip-relations", action="store_true", help="Skip relation extraction")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip scoring step")
    parser.add_argument("--skip-report", action="store_true", help="Skip report generation")
    parser.add_argument("--no-save-data", action="store_true", help="Skip saving processed data")
    parser.add_argument("--scoring-types", nargs='+', 
                       default=['aging_relevance', 'bridge_candidates'],
                       help="Scoring types to run")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Prepare options
        options = {
            'ner_method': args.ner_method,
            'skip_relations': args.skip_relations,
            'skip_scoring': args.skip_scoring,
            'skip_report': args.skip_report,
            'save_data': not args.no_save_data,
            'scoring_types': args.scoring_types,
        }
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting HALDxAI pipeline in {args.mode} mode...")
        logger.info(f"Input: {args.input}")
        logger.info(f"Output: {args.output}")
        
        # Run pipeline based on mode
        success = False
        
        if args.mode == 'complete':
            success = run_complete_pipeline(config, args.input, args.output, options)
        elif args.mode == 'ner':
            success = run_ner_only(config, args.input, args.output, options)
        elif args.mode == 'analysis':
            success = run_analysis_only(config, args.input, args.output, options)
        
        if success:
            logger.info("Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("Pipeline failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()