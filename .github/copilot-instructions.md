This document provides essential guidance for AI agents working on the Lyra codebase. Lyra integrates a Graph Neural Network (GNN) as a differentiable, episodic memory layer directly within a Gemma LLM.

## 1. Core Architecture

The central concept is a **Unified Latent Space (ULS)** where Gemma's text processing and the GNN's memory operations are fused. This is achieved by inserting a custom `MemoryInjectionLayer` between the layers of a Gemma model.

- **Base Model**: A Gemma 1B model. During the initial training of the memory components, the base model's weights are frozen.
- **Episodic Memory**: A GNN (`lyra/gnn.py`) stores "episodes" (e.g., sentences) as nodes in a graph.
- **Integration**: The `MemoryInjectionLayer` (`lyra/injection.py`) acts as the bridge. It is inserted between two of Gemma's transformer blocks.

### Data Flow:

1.  Input text is processed by the initial layers of the Gemma model, producing hidden states.
2.  The `MemoryInjectionLayer` receives these hidden states.
3.  It pools the hidden states to create a `query_vector`.
4.  This query is sent to the `EpisodicMemoryGNN`.
5.  The GNN processes its graph and returns a single `episodic_context_embedding`.
6.  This embedding is projected and added back to the hidden states.
7.  The memory-enriched hidden states are passed to the remaining Gemma layers for final processing.

The main model orchestrating this is `GemmaWithMemory` in `lyra/model.py`.

## 2. Key Files & Modules

-   `lyra/model.py`: Defines `GemmaWithMemory`, the main integrated model class that inherits from Hugging Face's `GemmaPreTrainedModel`. This is the primary entry point for the combined model.
-   `lyra/gnn.py`: Contains the `EpisodicMemoryGNN` (built with PyTorch Geometric). This is the memory store.
-   `lyra/injection.py`: Contains the `MemoryInjectionLayer`. This is the critical integration component where the GNN is queried and its output is fused with the LLM's state.
-   `scripts/train.py`: The training script. **Crucially, the optimizer here is configured to train *only* the parameters of the GNN and the injection layer.** The base Gemma model remains frozen during this phase.
-   `scripts/generate_dataset.py`: Creates the synthetic, memory-dependent Q&A data needed for fine-tuning.
-   `scripts/usage_example.py`: Demonstrates the intended stateful inference workflow.

## 3. Developer Workflows

### Training

The training process is non-standard.

-   **Data Loading**: The `DataLoader` uses a custom `collate_fn` to construct the `torch_geometric.data.Data` object (the memory graph) for each training sample.
-   **Model Forward Pass**: The training loop passes both the tokenized question and the corresponding `memory_graph` to the `GemmaWithMemory.forward` method.
-   **Optimization**: The optimizer (e.g., AdamW) is explicitly given only the trainable parameters: `list(model.gnn.parameters()) + list(model.injection_layer.parameters())`.

### Stateful Inference

The `GemmaWithMemory` model is stateful. The intended usage pattern is:

1.  Load the fine-tuned model.
2.  Clear any previous memory: `model.clear_memory()`.
3.  Add context sentences one by one to build the internal memory graph: `model.add_to_memory("Fact 1...")`.
4.  Ask a question that requires memory: `model.generate("What is fact 1?")`.

The `generate` method works out-of-the-box because the `forward` pass is designed to use the model's internal `self.memory_graph` when no explicit graph is provided.

## 4. Project-Specific Conventions

-   **Hugging Face Compatibility**: The `GemmaWithMemory` class is designed to be compatible with the Hugging Face `transformers` library, particularly for saving, loading (`save_pretrained`, `from_pretrained`), and generation (`generate`). The custom `forward` method signature is carefully crafted for this purpose.
-   **Memory Management**: The external API for managing memory (`add_to_memory`, `clear_memory`) is the primary way to interact with the model's state during inference.


## 2. Key Files & Modules

-   `lyra/model.py`: Defines `GemmaWithMemory`, the main integrated model class that inherits from Hugging Face's `GemmaPreTrainedModel`. This is the primary entry point for the combined model.
-   `lyra/gnn.py`: Contains the `EpisodicMemoryGNN` (built with PyTorch Geometric). This is the memory store.
-   `lyra/injection.py`: Contains the `MemoryInjectionLayer`. This is the critical integration component where the GNN is queried and its output is fused with the LLM's state.
-   `scripts/train.py`: The training script. **Crucially, the optimizer here is configured to train *only* the parameters of the GNN and the injection layer.** The base Gemma model remains frozen.
-   `scripts/generate_dataset.py`: Creates the synthetic, memory-dependent Q&A data needed for fine-tuning.
-   `scripts/usage_example.py`: Demonstrates the intended stateful inference workflow.

## 3. Developer Workflows

### Training

The training process is non-standard.

-   **Data Loading**: The `DataLoader` uses a custom `collate_fn` to construct the `torch_geometric.data.Data` object (the memory graph) for each training sample.
-   **Model Forward Pass**: The training loop passes both the tokenized question and the corresponding `memory_graph` to the `GemmaWithMemory.forward` method.
-   **Optimization**: The optimizer (e.g., AdamW) is explicitly given only the trainable parameters: `list(model.gnn.parameters()) + list(model.injection_layer.parameters())`.

### Stateful Inference

The `GemmaWithMemory` model is stateful. The intended usage pattern is:

1.  Load the fine-tuned model.
2.  Clear any previous memory: `model.clear_memory()`.
3.  Add context sentences one by one to build the internal memory graph: `model.add_to_memory("Fact 1...")`.
4.  Ask a question that requires memory: `model.generate("What is fact 1?")`.

The `generate` method works out-of-the-box because the `forward` pass is designed to use the model's internal `self.memory_graph` when no explicit graph is provided.

## 4. Project-Specific Conventions

-   **Frozen Core for Adapter Training**: When developing the memory components, the initial training phase is performed against a frozen Gemma model. This isolates the learning to the GNN and injection layer. However, the final, merged model is not permanently frozen.
-   **Hugging Face Compatibility**: The `GemmaWithMemory` class is designed to be compatible with the Hugging Face `transformers` library, particularly for saving, loading (`save_pretrained`, `from_pretrained`), and generation (`generate`). The custom `forward` method signature is carefully crafted for this purpose.
-   **Memory Management**: The external API for managing memory (`add_to_memory`, `clear_memory`) is the primary way to interact with the model's state during inference.
