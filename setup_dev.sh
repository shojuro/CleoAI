#!/bin/bash
# Development Environment Setup Script for CleoAI
# This script sets up a complete development environment

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_info() {
    echo -e "${YELLOW}➜ $1${NC}"
}

# Check Python version
check_python() {
    print_info "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
        PYTHON_CMD="python"
    else
        print_error "Python is not installed!"
        exit 1
    fi
    
    # Check if Python version is 3.8 or higher
    if [ "$(echo "$PYTHON_VERSION 3.8" | awk '{print ($1 >= $2)}')" -eq 1 ]; then
        print_success "Python $PYTHON_VERSION found"
    else
        print_error "Python 3.8 or higher is required (found $PYTHON_VERSION)"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_info "Creating virtual environment..."
    
    if [ -d "venv" ]; then
        print_info "Virtual environment already exists. Skipping creation."
    else
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    fi
    
    # Activate virtual environment
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    print_success "Virtual environment activated"
}

# Upgrade pip
upgrade_pip() {
    print_info "Upgrading pip..."
    pip install --upgrade pip wheel setuptools
    print_success "pip upgraded"
}

# Install dependencies
install_dependencies() {
    print_info "Installing core dependencies..."
    pip install -r requirements.txt
    print_success "Core dependencies installed"
    
    print_info "Installing development dependencies..."
    pip install -r requirements-dev.txt
    print_success "Development dependencies installed"
}

# Setup pre-commit hooks
setup_precommit() {
    print_info "Setting up pre-commit hooks..."
    pre-commit install
    print_success "Pre-commit hooks installed"
    
    print_info "Running pre-commit on all files..."
    pre-commit run --all-files || true
    print_success "Pre-commit initial run completed"
}

# Create necessary directories
create_directories() {
    print_info "Creating project directories..."
    
    directories=(
        "data/raw"
        "data/processed"
        "models/checkpoints"
        "outputs/evaluations"
        "logs/training"
        "docs/_build"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    print_success "Project directories created"
}

# Check CUDA availability
check_cuda() {
    print_info "Checking CUDA availability..."
    
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        print_success "CUDA is available"
    else
        print_info "CUDA not found. GPU acceleration will not be available."
    fi
}

# Run initial tests
run_tests() {
    print_info "Running initial tests..."
    
    # Run linting
    print_info "Running linters..."
    flake8 src tests --count --statistics || true
    
    # Run type checking
    print_info "Running type checker..."
    mypy src || true
    
    # Run unit tests
    print_info "Running unit tests..."
    pytest tests/unit -v --tb=short || true
    
    print_success "Initial tests completed"
}

# Download required models (optional)
download_models() {
    print_info "Would you like to download the base model? (y/n)"
    read -r response
    
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_info "Downloading base model..."
        python -c "
from transformers import AutoTokenizer, AutoModel
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
print('Tokenizer downloaded successfully')
"
        print_success "Model components downloaded"
    else
        print_info "Skipping model download"
    fi
}

# Generate initial documentation
generate_docs() {
    print_info "Generating initial documentation..."
    
    # Create docs configuration if it doesn't exist
    if [ ! -f "docs/conf.py" ]; then
        mkdir -p docs
        cat > docs/conf.py << 'EOF'
# Configuration file for the Sphinx documentation builder.
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'CleoAI'
copyright = '2024, CleoAI Team'
author = 'CleoAI Team'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_rtd_theme',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
EOF
    fi
    
    # Create index.rst if it doesn't exist
    if [ ! -f "docs/index.rst" ]; then
        cat > docs/index.rst << 'EOF'
Welcome to CleoAI's documentation!
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
EOF
    fi
    
    print_success "Documentation structure created"
}

# Main setup flow
main() {
    echo "========================================="
    echo "   CleoAI Development Environment Setup"
    echo "========================================="
    echo
    
    check_python
    create_venv
    upgrade_pip
    install_dependencies
    setup_precommit
    create_directories
    check_cuda
    generate_docs
    download_models
    run_tests
    
    echo
    echo "========================================="
    print_success "Development environment setup complete!"
    echo "========================================="
    echo
    print_info "Next steps:"
    echo "  1. Activate the virtual environment:"
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        echo "     source venv/Scripts/activate"
    else
        echo "     source venv/bin/activate"
    fi
    echo "  2. Run the test suite:"
    echo "     pytest"
    echo "  3. Start developing!"
    echo "     python main.py --help"
    echo
}

# Run main function
main