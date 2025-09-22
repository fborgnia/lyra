# Lyra Codebase Guide for AI Agents

This document provides essential information for navigating and contributing to the Lyra codebase. Lyra implements `GemmaWithMemory`, a model that enhances a standard Gemma instruction-tuned model with an internal episodic memory graph.

## Architecture Overview

The core of the project is the `GemmaWithMemory` model, located in `lyra/model.py`. This model wraps a frozen Gemma instruction-tuned model and adds an internal, GNN-based episodic memory.

Key components:
- **`lyra/model.py`**: Contains the `GemmaWithMemory` class, which is the main model implementation. It integrates the base Gemma model with a GNN and an injection layer to manage memory.
- **`lyra/gnn.py`**: Defines the Graph Neural Network used for processing the memory graph.
- **`lyra/injection.py`**: Implements the layer that injects memory-related information into the Gemma model.
- **`lyra/dataset.py`**: Handles data loading and preparation for training.

### Memory Management

The memory management is designed to be implicit and transparent, triggered by Gemma's special turn tokens:
- **Memory Update**: The memory is updated after for every user call to the generate api. The model creates a pooled vector from the user input and adds it as a new node to the memory graph.

There are no public methods to control the memory; it is all handled internally by the model.

## Developer Workflows

### 1. Dataset Generation

To generate the synthetic dataset for training, use the script in `scripts/generate_dataset.py`.

### 2. Training

The training process is divided into two main stages:

- **Stage 1: Semantic Alignment**: Train the **GNN itself** to refine raw memory vectors into meaningful, semantically-aligned node embeddings. This is done using a triplet loss function.
- **Stage 2: Retrieval Skill Training**: Fine-tune the model on conversational data. The model learns to use its memory implicitly to improve its responses.

The main training script is `scripts/train.py`.

### 3. Usage Example

To see an example of how to use the `GemmaWithMemory` model for inference, refer to `scripts/usage_example.py`. This script demonstrates how to load the model and interact with it in a conversational manner.

## Key Files and Directories

- `lyra/`: The main source code for the Lyra model and its components.
- `scripts/`: Contains scripts for training, data generation, and usage examples.
- `data/`: Stores datasets, including the synthetic dataset generated for training.
- `models/`: Contains the pre-trained Gemma model used as the base for `GemmaWithMemory`.
- `tests/`: Unit tests for the different components of the model.
