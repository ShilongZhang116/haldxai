# -*- coding: utf-8 -*-
"""
Command-line interface for HALDxAI platform.
"""

import sys
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .core.pipeline import HALDxAI, PipelineConfig
from .config import Settings, load_config, save_config
from .utils import setup_logging

app = typer.Typer(
    name="haldxai",
    help="HALDxAI: Healthy Aging and Longevity Discovery AI Platform",
    add_completion=False,
)

console = Console()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v", help="Show version and exit"
    ),
    config: Optional[str] = typer.Option(
        None, "--config", "-c", help="Configuration file path"
    ),
    verbose: bool = typer.Option(
        False, "--verbose", help="Enable verbose logging"
    ),
):
    """HALDxAI: Healthy Aging and Longevity Discovery AI Platform"""
    if version:
        from . import __version__
        console.print(f"HALDxAI version {__version__}")
        raise typer.Exit()
    
    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(log_level)


@app.command()
def init(
    project_path: str = typer.Argument(".", help="Project path to initialize"),
    force: bool = typer.Option(False, "--force", help="Force overwrite existing files"),
):
    """Initialize a new HALDxAI project."""
    try:
        project_root = Path(project_path).resolve()
        console.print(f"🚀 Initializing HALDxAI project at: {project_root}")
        
        # Create directory structure
        directories = [
            "data/raw",
            "data/processed", 
            "data/outputs",
            "models",
            "logs",
            "configs",
            "notebooks",
            "tests",
        ]
        
        for dir_path in directories:
            full_path = project_root / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            console.print(f"📁 Created directory: {dir_path}")
        
        # Create default configuration
        config_path = project_root / "configs" / "config.yaml"
        if not config_path.exists() or force:
            settings = Settings()
            settings.paths.project_root = str(project_root)
            save_config(settings, config_path)
            console.print(f"⚙️  Created configuration: {config_path}")
        
        # Create .env file
        env_path = project_root / ".env"
        from .config.defaults import DEFAULT_ENV
        if not env_path.exists() or force:
            env_path.write_text(DEFAULT_ENV.strip())
            console.print(f"🔑 Created environment file: {env_path}")
        
        # Create .gitignore
        gitignore_path = project_root / ".gitignore"
        if not gitignore_path.exists() or force:
            gitignore_content = """
# Data
data/raw/*
data/processed/*
data/outputs/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/outputs/.gitkeep

# Models
models/*
!models/.gitkeep

# Logs
logs/*
!logs/.gitkeep

# Environment
.env

# Cache
.cache/
__pycache__/
*.pyc

# IDE
.vscode/
.idea/

# Python
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
            """.strip()
            gitignore_path.write_text(gitignore_content)
            console.print(f"📝 Created .gitignore file")
        
        console.print(Panel(
            f"✅ HALDxAI project initialized successfully!\n\n"
            f"Next steps:\n"
            f"1. Edit {env_path} with your API keys\n"
            f"2. Edit {config_path} for your settings\n"
            f"3. Run 'haldxai process' to start analysis",
            title="Success",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"❌ Error initializing project: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def process(
    input_file: str = typer.Argument(..., help="Input file to process"),
    output_dir: str = typer.Option("outputs", "--output", "-o", help="Output directory"),
    ner_method: str = typer.Option("llm", "--ner", help="NER method (llm/spacy)"),
    extract_relations: bool = typer.Option(True, "--relations/--no-relations", help="Extract relations"),
    run_scoring: bool = typer.Option(True, "--scoring/--no-scoring", help="Run scoring"),
    generate_report: bool = typer.Option(True, "--report/--no-report", help="Generate report"),
):
    """Process input file through the complete HALDxAI pipeline."""
    try:
        console.print("🔄 Starting HALDxAI processing pipeline...")
        
        # Initialize HALDxAI
        hald = HALDxAI()
        
        # Step 1: NER
        console.print("📝 Step 1: Running Named Entity Recognition...")
        ner_results = hald.run_ner_pipeline(input_file, output_dir, ner_method)
        if not ner_results.success:
            console.print(f"❌ NER failed: {ner_results.message}", style="red")
            raise typer.Exit(1)
        console.print(f"✅ NER completed: {ner_results.message}")
        
        # Step 2: Relation extraction
        if extract_relations:
            console.print("🔗 Step 2: Extracting relations...")
            rel_results = hald.run_relation_extraction(
                f"{output_dir}/entities.json", 
                output_dir
            )
            if not rel_results.success:
                console.print(f"❌ Relation extraction failed: {rel_results.message}", style="red")
                raise typer.Exit(1)
            console.print(f"✅ Relation extraction completed: {rel_results.message}")
        
        # Step 3: Scoring
        if run_scoring:
            console.print("📊 Step 3: Running scoring...")
            score_results = hald.run_scoring_pipeline(output_dir)
            if not score_results.success:
                console.print(f"❌ Scoring failed: {score_results.message}", style="red")
                raise typer.Exit(1)
            console.print(f"✅ Scoring completed: {score_results.message}")
        
        # Step 4: Build knowledge graph
        console.print("🕸️  Step 4: Building knowledge graph...")
        kg_results = hald.build_knowledge_graph(output_dir=output_dir)
        if not kg_results.success:
            console.print(f"❌ Knowledge graph construction failed: {kg_results.message}", style="red")
            raise typer.Exit(1)
        console.print(f"✅ Knowledge graph built: {kg_results.message}")
        
        # Step 5: Generate report
        if generate_report:
            console.print("📄 Step 5: Generating report...")
            report_results = hald.generate_report(f"{output_dir}/report.html")
            if not report_results.success:
                console.print(f"❌ Report generation failed: {report_results.message}", style="red")
                raise typer.Exit(1)
            console.print(f"✅ Report generated: {report_results.message}")
        
        console.print(Panel(
            "🎉 Processing pipeline completed successfully!\n\n"
            f"Check the output directory: {output_dir}",
            title="Success",
            border_style="green"
        ))
        
    except Exception as e:
        console.print(f"❌ Processing failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def analyze(
    data_dir: str = typer.Argument(..., help="Directory containing processed data"),
    output_dir: str = typer.Option("analysis", "--output", "-o", help="Output directory"),
):
    """Analyze processed data and generate insights."""
    try:
        console.print("🔍 Starting data analysis...")
        
        # Initialize HALDxAI and load data
        hald = HALDxAI()
        load_results = hald.load_data(data_dir)
        if not load_results.success:
            console.print(f"❌ Failed to load data: {load_results.message}", style="red")
            raise typer.Exit(1)
        
        # Get statistics
        stats = hald.get_statistics()
        
        # Display statistics table
        console.print("\n📊 Entity Statistics:")
        entity_table = Table()
        entity_table.add_column("Metric", style="cyan")
        entity_table.add_column("Value", style="green")
        
        entity_stats = stats["entities"]
        entity_table.add_row("Total Entities", str(entity_stats["total_entities"]))
        entity_table.add_row("Total Synonyms", str(entity_stats["total_synonyms"]))
        entity_table.add_row("Average Confidence", f"{entity_stats['average_confidence']:.3f}")
        
        console.print(entity_table)
        
        console.print("\n🔗 Relation Statistics:")
        relation_table = Table()
        relation_table.add_column("Metric", style="cyan")
        relation_table.add_column("Value", style="green")
        
        relation_stats = stats["relations"]
        relation_table.add_row("Total Relations", str(relation_stats["total_relations"]))
        relation_table.add_row("Directed Relations", str(relation_stats["directed_relations"]))
        relation_table.add_row("Connected Entities", str(relation_stats["connected_entities"]))
        relation_table.add_row("Average Confidence", f"{relation_stats['average_confidence']:.3f}")
        
        console.print(relation_table)
        
        # Generate analysis report
        report_results = hald.generate_report(f"{output_dir}/analysis_report.html")
        if report_results.success:
            console.print(f"✅ Analysis report generated: {output_dir}/analysis_report.html")
        
    except Exception as e:
        console.print(f"❌ Analysis failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    validate: bool = typer.Option(False, "--validate", help="Validate configuration"),
    key: Optional[str] = typer.Option(None, "--key", help="Configuration key to set"),
    value: Optional[str] = typer.Option(None, "--value", help="Configuration value to set"),
):
    """Manage configuration settings."""
    try:
        settings = load_config()
        
        if show:
            config_dict = settings.to_dict()
            console.print(Panel(
                str(config_dict),
                title="Current Configuration",
                border_style="blue"
            ))
        
        elif validate:
            from .config.settings import validate_config
            issues = validate_config(settings)
            if issues:
                console.print("❌ Configuration issues found:", style="red")
                for issue in issues:
                    console.print(f"  • {issue}", style="red")
            else:
                console.print("✅ Configuration is valid", style="green")
        
        elif key and value:
            # Simple key-value setting (basic implementation)
            if hasattr(settings, key):
                setattr(settings, key, value)
                console.print(f"✅ Set {key} = {value}")
            else:
                console.print(f"❌ Unknown configuration key: {key}", style="red")
        
        else:
            console.print("Use --show, --validate, or --key/--value to manage configuration")
    
    except Exception as e:
        console.print(f"❌ Configuration management failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def server(
    host: str = typer.Option("localhost", "--host", help="Server host"),
    port: int = typer.Option(8000, "--port", help="Server port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
):
    """Start the HALDxAI web server."""
    try:
        console.print(f"🌐 Starting HALDxAI server on {host}:{port}")
        
        # This would be implemented with FastAPI
        console.print("🚧 Web server feature coming soon!", style="yellow")
        
    except Exception as e:
        console.print(f"❌ Server startup failed: {e}", style="red")
        raise typer.Exit(1)


def main_cli():
    """Main CLI entry point."""
    try:
        app()
    except KeyboardInterrupt:
        console.print("\n👋 Goodbye!", style="yellow")
        sys.exit(0)
    except Exception as e:
        console.print(f"❌ Unexpected error: {e}", style="red")
        sys.exit(1)


if __name__ == "__main__":
    main_cli()