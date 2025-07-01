.. CleoAI documentation master file

Welcome to CleoAI's documentation!
==================================

CleoAI is an advanced autonomous AI agent system built with PyTorch and Transformers, 
featuring a Mixture of Experts (MoE) architecture and sophisticated memory management capabilities.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   architecture
   memory
   training
   inference
   examples
   contributing

Overview
--------

CleoAI provides:

* **Mixture of Experts Architecture**: Efficient expert routing for specialized task handling
* **Advanced Memory Systems**: Short-term, long-term, episodic, and procedural memory
* **Distributed Training**: DeepSpeed integration for efficient large-scale training
* **Flexible Inference**: Support for streaming and batch inference
* **Comprehensive CLI**: Full-featured command-line interface

Key Features
------------

**Model Architecture**
   - Mixture of Experts (MoE) with configurable routing
   - Support for both dense and sparse models
   - Efficient memory usage through conditional computation

**Memory Management**
   - Short-term conversation memory with recency weighting
   - Long-term user preference storage
   - Episodic memory for contextual recall
   - Procedural memory for task patterns

**Training Pipeline**
   - Multi-phase training support
   - Distributed training with DeepSpeed
   - Automatic checkpointing and recovery
   - Comprehensive evaluation metrics

**Inference Engine**
   - Optimized text generation
   - Streaming response support
   - Context-aware generation
   - Memory-augmented responses

Getting Started
---------------

To get started with CleoAI:

1. Install the package::

    pip install -r requirements.txt

2. Run a simple inference::

    python main.py interactive --model ./models/cleoai

3. Train a new model::

    python main.py train --model mistralai/Mistral-7B-v0.1

For more detailed instructions, see the :doc:`installation` and :doc:`quickstart` guides.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`