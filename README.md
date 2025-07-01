# CleoAI - Autonomous AI Agent System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

CleoAI is an advanced autonomous AI agent system built with PyTorch and Transformers, featuring a Mixture of Experts (MoE) architecture and sophisticated memory management capabilities. Designed for conversational AI applications with focus on personalization and long-term memory retention.

## 🚀 Features

- **Mixture of Experts (MoE) Architecture**: Efficient expert routing for specialized task handling
- **Advanced Memory Systems**: 
  - Short-term conversational memory
  - Long-term user preference storage
  - Episodic memory for contextual recall
  - Procedural memory for task execution patterns
- **Distributed Memory Backends**:
  - Redis for high-speed caching
  - MongoDB for document archival
  - Supabase for real-time sync
  - Pinecone for vector search
  - Automatic tiered storage management
- **Production-Ready Infrastructure**:
  - GraphQL API with health monitoring
  - Docker Compose for easy deployment
  - Configuration validation
  - Memory migration tools
- **Distributed Training**: DeepSpeed integration for efficient large-scale training
- **Flexible Inference**: Support for streaming and batch inference
- **Comprehensive CLI**: Full-featured command-line interface for training, inference, and evaluation

## 📋 Requirements

- Python 3.8 or higher
- CUDA 11.8+ (for GPU support)
- 16GB+ RAM (32GB+ recommended)
- 50GB+ free disk space

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/CleoAI.git
cd CleoAI
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install core dependencies
pip install -r requirements.txt

# Install development dependencies (for contributing)
pip install -r requirements-dev.txt
```

### 4. Set Up Pre-commit Hooks (for development)

```bash
pre-commit install
```

### 5. Configure Environment

```bash
# Copy environment template
cp .env.template .env

# Edit .env with your settings
# For local development, the defaults work out of the box
```

### 6. Verify Installation

```bash
python main.py --help

# Check configuration validity
python -c "from src.utils.config_validator import ensure_valid_configuration; ensure_valid_configuration()"
```

## 🚦 Quick Start

### Training a Model

```bash
# Basic training
python main.py train --model mistralai/Mistral-7B-v0.1 --output-dir ./models/my_model

# Training with specific configuration
python main.py train \
    --model mistralai/Mistral-7B-v0.1 \
    --output-dir ./models/my_model \
    --num-experts 8 \
    --use-moe \
    --deepspeed
```

### Running Inference

```bash
# Single inference
python main.py infer \
    --model ./models/my_model \
    --input "Hello, how are you?" \
    --temperature 0.7

# Interactive mode
python main.py interactive \
    --model ./models/my_model \
    --temperature 0.7
```

### Evaluating a Model

```bash
python main.py evaluate \
    --model ./models/my_model \
    --dataset all \
    --output-dir ./outputs/evaluation
```

## 🗄️ Distributed Memory Setup

CleoAI now supports distributed memory backends for production scalability:

### Quick Setup with Docker

```bash
# Start all services locally
docker-compose up -d

# Check service health
curl http://localhost:8000/health/detailed
```

### Available Backends

- **Redis** - High-speed cache for active conversations
- **MongoDB** - Document storage for archival data  
- **Supabase** - PostgreSQL with real-time sync
- **Pinecone** - Vector database for semantic search
- **SQLite** - Local storage (legacy/development)
- **ChromaDB** - Local vector storage (legacy)

### Configuration

Edit `.env` to enable/disable backends:

```env
# Enable distributed backends
USE_REDIS=true
USE_MONGODB=true
USE_SUPABASE=true  # Requires API keys
USE_PINECONE=true  # Requires API keys

# Keep legacy for migration
USE_SQLITE=true
USE_CHROMADB=true
```

### Starting the API Server

```bash
# Start with GraphQL API
python main.py api

# Start with debug mode
python main.py api --debug

# Access endpoints
# GraphQL: http://localhost:8000/graphql
# Health: http://localhost:8000/health
```

### Memory Migration

Migrate from SQLite to distributed backends:

```bash
# Dry run first
python scripts/migrate_memory.py --source sqlite --target redis,mongodb --dry-run

# Run migration
python scripts/migrate_memory.py --source sqlite --target redis,mongodb,supabase,pinecone
```

### Backend Selection Guide

See [docs/backend_selection_guide.md](docs/backend_selection_guide.md) for detailed recommendations.

**Quick Recommendations:**
- **Development**: SQLite + ChromaDB
- **Small Production**: Redis + MongoDB
- **Large Production**: All distributed backends

## 📁 Project Structure

```
CleoAI/
├── config.py                 # Configuration classes
├── main.py                   # Main CLI entry point
├── requirements.txt          # Core dependencies
├── requirements-dev.txt      # Development dependencies
├── src/
│   ├── model/
│   │   └── moe_model.py     # MoE model implementation
│   ├── memory/
│   │   ├── memory_manager.py # Memory management system
│   │   └── enhanced_shadow_memory.py
│   ├── training/
│   │   └── trainer.py       # Training pipeline
│   └── inference/
│       └── inference_engine.py # Inference engine
├── tests/                   # Test suite
├── docs/                    # Documentation
└── configs/                 # Configuration files
```

## ⚙️ Configuration

The system uses a hierarchical configuration system defined in `config.py`:

- **ModelConfig**: Model architecture settings
- **MemoryConfig**: Memory system parameters
- **TrainingConfig**: Training hyperparameters
- **EvaluationConfig**: Evaluation settings
- **ProjectConfig**: Project paths and directories

### Custom Configuration

Create a custom config file:

```python
from config import model_config, training_config

# Modify configurations
model_config.num_experts = 16
model_config.hidden_size = 4096

training_config.learning_rate = 1e-5
training_config.batch_size = 8
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m unit        # Unit tests only
pytest -m integration # Integration tests only
```

## 🔧 Development

### Code Quality

```bash
# Format code
black .

# Lint code
flake8 src tests

# Type checking
mypy src

# All checks (via pre-commit)
pre-commit run --all-files
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- All tests pass
- Code follows the project style guide
- Documentation is updated
- Commit messages are clear and descriptive

## 📊 Performance

### Hardware Requirements

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| CPU | 4 cores | 8 cores | 16+ cores |
| RAM | 16GB | 32GB | 64GB+ |
| GPU | GTX 1080 (8GB) | RTX 3090 (24GB) | A100 (40GB+) |
| Storage | 50GB SSD | 200GB SSD | 500GB+ NVMe |

### Training Performance

- **Single GPU**: ~1000 tokens/second on RTX 3090
- **Multi-GPU**: Near-linear scaling with DeepSpeed
- **Memory Usage**: ~20GB for 7B parameter model with MoE

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python main.py train --batch-size 2 --gradient-accumulation-steps 16
   ```

2. **Import Errors**
   ```bash
   # Ensure virtual environment is activated
   source venv/bin/activate
   # Reinstall dependencies
   pip install -r requirements.txt
   ```

3. **Slow Training**
   ```bash
   # Enable mixed precision
   python main.py train --fp16
   # Use DeepSpeed
   python main.py train --deepspeed
   ```

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Training Guide](docs/training.md)
- [Memory System Guide](docs/memory.md)
- [Deployment Guide](docs/deployment.md)

## 🔒 Security

- Never commit sensitive data or API keys
- Use environment variables for secrets
- Keep dependencies updated
- Report security issues to security@cleoai.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [DeepSpeed](https://github.com/microsoft/DeepSpeed)
- [ChromaDB](https://www.trychroma.com/)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/CleoAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/CleoAI/discussions)
- **Email**: support@cleoai.com

---

**Note**: This project is under active development. Features and APIs may change. Please refer to the [CHANGELOG](CHANGELOG.md) for version-specific information.