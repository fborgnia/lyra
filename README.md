# Lyra: A Gemma-based Language Model with Episodic Memory

Lyra is an experimental language model that enhances a standard Gemma instruction-tuned model with an internal episodic memory graph. This allows the model to maintain a persistent, structured memory of conversations, leading to more context-aware and coherent interactions over extended periods.

## Architecture Overview

The core of the project is the `GemmaWithMemory` model, located in `lyra/model.py`. This model subclasses `transformers.Gemma3ForCausalLM` to integrate a Graph Neural Network (GNN) for managing an episodic memory graph.

### Key Components:

- **`lyra/model.py`**: Contains the `GemmaWithMemory` class. This class overrides the `generate()` and `forward()` methods of the base Gemma model to implement memory operations.
  - **`generate()` (Inference)**: For each user prompt, this method first updates the memory graph with a representation of the prompt, then queries the GNN for relevant past memories, injects them into the prompt embeddings, and finally calls the original `super().generate()` to produce a response.
  - **`forward()` (Training)**: This method is designed for training the GNN. It takes conversational data, simulates the memory-building process, and computes a loss that allows the GNN's parameters to be fine-tuned.
- **`lyra/gnn.py`**: Defines the `EpisodicMemoryGNN`, a GNN that performs attention-based retrieval over the memory graph to find nodes relevant to a given query.
- **`lyra/injection.py`**: Implements the `MemoryInjectionLayer`, which is responsible for prepending the retrieved memory context to the input embeddings before they are processed by the main language model.
- **`lyra/dataset.py`**: Handles data loading and preparation for training the GNN.

### Memory Management

Memory management is handled implicitly within the `generate()` method:

1.  **Memory Capture**: After receiving a user prompt, the model computes its embeddings and then creates a single pooled vector representing the "gist" of that prompt.
2.  **Memory Storage**: This vector is added as a new node to the internal `memory_graph`.
3.  **Memory Retrieval**: The pooled vector is also used as a query to the GNN, which retrieves a context vector summarizing relevant past memories.
4.  **Memory Injection**: The retrieved context is prepended to the prompt's embeddings before being passed to the Gemma model for generation.

This entire process is transparent to the user and happens as a pre-computation step within each `generate()` call.

## Developer Workflows

### 1. Dataset Generation

To generate a synthetic dataset for training the GNN, use the script in `scripts/generate_dataset.py`.

```bash
python scripts/generate_dataset.py
```

### 2. Training

The main training script is `scripts/train.py`. It fine-tunes the GNN on conversational data, teaching it to retrieve relevant memories.

```bash
python scripts/train.py
```

### 3. Usage Example

To see an example of how to use the `GemmaWithMemory` model for inference, refer to `scripts/usage_example.py`. This script demonstrates how to load the model and interact with it.

```bash
python scripts/usage_example.py
```

## Key Files and Directories

-   `lyra/`: Main source code for the Lyra model and its components.
-   `scripts/`: Scripts for training, data generation, and usage examples.
-   `data/`: Stores datasets.
-   `models/`: Contains the pre-trained Gemma model.
-   `tests/`: Unit tests.
-   `docs/`: Detailed documentation on the model's architecture.
