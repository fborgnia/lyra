# Lyra: A Gemma-based LLM with In-Layer Episodic Memory

Lyra is an experimental language model that enhances a standard Gemma instruction-tuned model with a novel, deeply integrated episodic memory system. Instead of managing context by manipulating prompts externally, Lyra injects memory directly into the residual stream of each Transformer decoder layer.

This allows the model to be aware of conversational history at every stage of processing, enabling a more nuanced form of context-awareness. The memory modules are designed to be lightweight and trainable via Parameter-Efficient Fine-Tuning (PEFT), leaving the base model's weights entirely frozen.

## Architecture Overview: How It Works

The core of Lyra's architecture is the in-place modification of Gemma's decoder layers. During model initialization, each `Gemma3DecoderLayer` is augmented with a `MemoryInjectionBlock`, turning it into a `LyraDecoderLayer`. This is achieved by dynamically replacing the `forward` method of each layer.

### Key Components:

-   **`lyra/model.py` (`Lyra`)**: The main model class that subclasses `Gemma3ForCausalLM`. It orchestrates the monkey-patching process and manages the `EpisodicMemoryStore`. It also overrides the `generate()` method to manage the archival of conversational turns.
-   **`lyra/memory/store.py` (`EpisodicMemoryStore`)**: A session-level cache that stores the `hidden_states` and `attention_mask` of past conversational turns as `MemoryPackage` objects.
-   **`lyra/memory/layer.py` (`LyraDecoderLayer`)**: Not a class that is instantiated, but a new `forward` method that is injected into each of Gemma's existing decoder layers. It contains the logic for integrating memory into the data path.
-   **`lyra/memory/injection.py` (`MemoryInjectionBlock`)**: The heart of the memory system. An instance of this block is created for **each decoder layer**. It is responsible for retrieving memories, projecting the current hidden state into a query, and performing cross-attention.
-   **`lyra/memory/attention.py` (`MemoryCrossAttention`)**: A trainable cross-attention module that calculates attention scores between the current layer's `query_states` and the `hidden_states` of past memories.

---

## The Data Flow: A Step-by-Step Guide

When `model.generate()` is called, the following happens at **each decoder layer**:

1.  **Normalization**: The layer's input `hidden_states` are passed through the standard `input_layernorm`.
2.  **Memory Branch Activation**:
    *   A residual connection (`memory_residual`) is created from the normalized `hidden_states`.
    *   The `MemoryInjectionBlock` for that specific layer is called.
3.  **Memory Injection**:
    *   The block retrieves relevant memories (e.g., the first and last turns) from the `EpisodicMemoryStore`.
    *   It uses its unique, layer-specific `q_proj` layer to project the current `hidden_states` into `query_states`.
    *   It calls the `MemoryCrossAttention` module, which attends to the selected past memories using the new `query_states`.
    *   The output is an `aggregated_memory_enrichment` tensor.
4.  **Residual Connection**:
    *   The `memory_output` is passed through its own `post_memory_layernorm`.
    *   The normalized memory is added back to the `memory_residual`.
5.  **Self-Attention**: This new, memory-infused `hidden_states` tensor is then passed to the original, **frozen** `self_attn` block of the Gemma layer.
6.  **Archival**: After the full model response is generated, the `Lyra` model takes the final `hidden_states` of the prompt-response pair and stores them in the `EpisodicMemoryStore` for use in subsequent turns.

---

## Developer Workflows

### 1. Stage 1 Training: Fine-Tuning the Memory Modules

The training script `scripts/train_memory_heads.py` is used to fine-tune the model. This script employs a PEFT strategy where all of the original Gemma model's parameters are frozen.

Only the newly added memory components are trained:
- The `q_proj` in each `MemoryInjectionBlock`.
- The `k_proj`, `v_proj`, and `o_proj` in the `MemoryCrossAttention` module.
- The `post_memory_layernorm` in each decoder layer.

This teaches the memory system how to generate outputs that the frozen self-attention and FFN layers can interpret, effectively learning to "speak Gemma's language."

```bash
# This script will train the new memory modules.
python3 scripts/train_memory_heads.py
```

### 2. Inference and Usage

To use the model, run an inference script like `scripts/test_cmd_1_turn.py`. The `Lyra` model will use its trained memory modules to provide context-aware responses across multiple turns.

```bash
# Run a multi-turn conversation to test the memory architecture.
python3 scripts/test_cmd_1_turn.py
```
