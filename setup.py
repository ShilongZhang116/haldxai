#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
HALDxAI Setup Script
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="haldxai",
    version="0.1.0",
    author="HALDxAI Development Team",
    author_email="haldxai@example.com",
    description="Healthy Aging and Longevity Discovery AI Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/HALDxAI-Repository",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "isort>=5.10.0",
            "mypy>=0.971",
            "pre-commit>=2.20.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "ipywidgets>=7.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "haldxai=haldxai.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "haldxai": [
            "configs/*.yaml",
            "data/*.json",
            "templates/*.html",
        ],
    },
    keywords="aging longevity bioinformatics nlp knowledge-graph machine-learning",
    project_urls={
        "Bug Reports": "https://github.com/your-org/HALDxAI-Repository/issues",
        "Source": "https://github.com/your-org/HALDxAI-Repository",
        "Documentation": "https://haldxai.readthedocs.io/",
    },
)