#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database setup script for HALDxAI platform.
"""

import sys
import argparse
from pathlib import Path
import psycopg2
from neo4j import GraphDatabase
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from haldxai.config import load_config
from haldxai.utils import setup_logging, get_logger


def setup_postgresql(config, drop_existing=False):
    """Setup PostgreSQL database with required tables."""
    logger = get_logger(__name__)
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(
            host=config.database.pg_host,
            port=config.database.pg_port,
            dbname="postgres",  # Connect to default database first
            user=config.database.pg_user,
            password=config.database.pg_pass
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Create database if it doesn't exist
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{config.database.pg_dbname}'")
        if not cursor.fetchone():
            cursor.execute(f"CREATE DATABASE {config.database.pg_dbname}")
            logger.info(f"Created PostgreSQL database: {config.database.pg_dbname}")
        
        # Close connection to default database
        cursor.close()
        conn.close()
        
        # Connect to target database
        conn = psycopg2.connect(
            host=config.database.pg_host,
            port=config.database.pg_port,
            dbname=config.database.pg_dbname,
            user=config.database.pg_user,
            password=config.database.pg_pass
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Create schema if it doesn't exist
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {config.database.schema}")
        
        # Drop existing tables if requested
        if drop_existing:
            logger.warning("Dropping existing tables...")
            cursor.execute(f"DROP SCHEMA IF EXISTS {config.database.schema} CASCADE")
            cursor.execute(f"CREATE SCHEMA {config.database.schema}")
        
        # Create tables
        tables = {
            "entities": f"""
                CREATE TABLE {config.database.schema}.entities (
                    id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    entity_type VARCHAR(50) NOT NULL,
                    description TEXT,
                    synonyms JSONB,
                    confidence FLOAT DEFAULT 0.0,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            "relations": f"""
                CREATE TABLE {config.database.schema}.relations (
                    id VARCHAR(255) PRIMARY KEY,
                    source_id VARCHAR(255) NOT NULL,
                    target_id VARCHAR(255) NOT NULL,
                    relation_type VARCHAR(50) NOT NULL,
                    description TEXT,
                    confidence FLOAT DEFAULT 0.0,
                    direction VARCHAR(20) DEFAULT 'directed',
                    weight FLOAT DEFAULT 1.0,
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """,
            
            "evidence": f"""
                CREATE TABLE {config.database.schema}.evidence (
                    id VARCHAR(255) PRIMARY KEY,
                    pmid VARCHAR(50) NOT NULL,
                    text TEXT NOT NULL,
                    confidence FLOAT DEFAULT 0.0,
                    source VARCHAR(100),
                    entity_id VARCHAR(255),
                    relation_id VARCHAR(255),
                    metadata JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        }
        
        for table_name, table_sql in tables.items():
            try:
                cursor.execute(table_sql)
                logger.info(f"Created table: {table_name}")
            except psycopg2.Error as e:
                if "already exists" in str(e):
                    logger.info(f"Table {table_name} already exists")
                else:
                    logger.error(f"Error creating table {table_name}: {e}")
                    raise
        
        # Create indexes
        indexes = [
            f"CREATE INDEX IF NOT EXISTS idx_entities_type ON {config.database.schema}.entities(entity_type)",
            f"CREATE INDEX IF NOT EXISTS idx_entities_name ON {config.database.schema}.entities(name)",
            f"CREATE INDEX IF NOT EXISTS idx_entities_confidence ON {config.database.schema}.entities(confidence)",
            f"CREATE INDEX IF NOT EXISTS idx_relations_source ON {config.database.schema}.relations(source_id)",
            f"CREATE INDEX IF NOT EXISTS idx_relations_target ON {config.database.schema}.relations(target_id)",
            f"CREATE INDEX IF NOT EXISTS idx_relations_type ON {config.database.schema}.relations(relation_type)",
            f"CREATE INDEX IF NOT EXISTS idx_relations_confidence ON {config.database.schema}.relations(confidence)",
            f"CREATE INDEX IF NOT EXISTS idx_evidence_pmid ON {config.database.schema}.evidence(pmid)",
            f"CREATE INDEX IF NOT EXISTS idx_evidence_entity ON {config.database.schema}.evidence(entity_id)",
            f"CREATE INDEX IF NOT EXISTS idx_evidence_relation ON {config.database.schema}.evidence(relation_id)",
        ]
        
        for index_sql in indexes:
            try:
                cursor.execute(index_sql)
                logger.info(f"Created index")
            except psycopg2.Error as e:
                logger.warning(f"Index creation warning: {e}")
        
        cursor.close()
        conn.close()
        
        logger.info("PostgreSQL setup completed successfully")
        
    except Exception as e:
        logger.error(f"PostgreSQL setup failed: {e}")
        raise


def setup_neo4j(config, drop_existing=False):
    """Setup Neo4j database with required constraints and indexes."""
    logger = get_logger(__name__)
    
    try:
        # Connect to Neo4j
        driver = GraphDatabase.driver(
            config.database.neo4j_uri,
            auth=(config.database.neo4j_user, config.database.neo4j_password)
        )
        
        with driver.session() as session:
            # Drop existing data if requested
            if drop_existing:
                logger.warning("Dropping existing Neo4j data...")
                session.run("MATCH (n) DETACH DELETE n")
                session.run("MATCH ()-[r]-() DELETE r")
            
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT relation_id IF NOT EXISTS FOR ()-[r:RELATES_TO]-() REQUIRE r.id IS UNIQUE",
            ]
            
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint: {constraint}")
                except Exception as e:
                    if "already exists" in str(e).lower():
                        logger.info(f"Constraint already exists: {constraint}")
                    else:
                        logger.warning(f"Constraint creation warning: {e}")
            
            # Create indexes
            indexes = [
                "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.entity_type)",
                "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX relation_type IF NOT EXISTS FOR ()-[r:RELATES_TO]-() ON (r.relation_type)",
            ]
            
            for index in indexes:
                try:
                    session.run(index)
                    logger.info(f"Created index: {index}")
                except Exception as e:
                    logger.warning(f"Index creation warning: {e}")
        
        driver.close()
        logger.info("Neo4j setup completed successfully")
        
    except Exception as e:
        logger.error(f"Neo4j setup failed: {e}")
        raise


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description="Setup HALDxAI databases")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--drop-postgres", action="store_true", help="Drop existing PostgreSQL tables")
    parser.add_argument("--drop-neo4j", action="store_true", help="Drop existing Neo4j data")
    parser.add_argument("--postgres-only", action="store_true", help="Setup PostgreSQL only")
    parser.add_argument("--neo4j-only", action="store_true", help="Setup Neo4j only")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level)
    logger = get_logger(__name__)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        logger.info("Starting database setup...")
        logger.info(f"PostgreSQL: {config.database.pg_host}:{config.database.pg_port}/{config.database.pg_dbname}")
        logger.info(f"Neo4j: {config.database.neo4j_uri}")
        
        # Setup databases
        if not args.neo4j_only:
            setup_postgresql(config, args.drop_postgres)
        
        if not args.postgres_only:
            setup_neo4j(config, args.drop_neo4j)
        
        logger.info("Database setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()