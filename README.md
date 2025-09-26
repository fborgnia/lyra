# Lyra: A Gemma-based Language Model with Episodic Memory

Lyra enhances a standard Gemma instruction-tuned model with an internal episodic memory cache. This allows the model to maintain a persistent, structured memory of conversations, enabling it to recall past information and provide context-aware responses without any external prompt management.

Lyra treats the first conversational turn as a **persistent instruction context**. This initial prompt sets the persona and objective for the entire session, ensuring the model's behavior remains consistent while subsequent turns are stored as retrievable episodic memories.

## Architecture Overview: How It Works

The core of the project is the `Lyra` model, which subclasses `transformers.Gemma3ForCausalLM`. It integrates a small, trainable `Retriever` module that acts as a semantic search engine over the conversational history.

### Key Components:

-   **`lyra/model.py` (`Lyra`)**: The main model class. It orchestrates the entire memory process, overriding the `generate()` method to intercept user prompts and manage the memory workflow.
-   **`lyra/retriever.py` (`Retriever`)**: A lightweight, trainable module. Its sole purpose is to find the most semantically relevant memories from the past. It takes a query vector (from the current prompt) and returns the **indices** of the best-matching memories.
-   **`lyra/injection.py` (`MemoryInjectionLayer`)**: The "prompt engineer." It takes the indices provided by the `Retriever`, retrieves the corresponding past conversation turns, and stitches them together with the current prompt to create a single, valid, multi-turn input for the base model.
-   **The Memory Buffer**: Implemented as a simple Python `list` within the `Lyra` class. Each item is a dictionary containing the semantic summary vector and the original `input_ids` of a conversational turn. The first entry (`memory_buffer[0]`) is treated as a special, persistent **instruction context** that is always injected into the prompt.

---

## The Inference Loop: A Step-by-Step Guide

When a user calls `model.generate()`:

1.  **Query**: The user's prompt is received.
2.  **Memory Retrieval (The Retriever's Job)**:
    *   The `Retriever` searches for the most relevant episodic memories by comparing the current prompt's vector against all memories **except for the first one** (the instruction).
    *   It returns the indices of the top `k` most relevant memories.
3.  **Prompt Engineering (The Injection Layer's Job)**:
    *   The `MemoryInjectionLayer` **always** retrieves the instruction context from `memory_buffer[0]`.
    *   It then retrieves the top `k` episodic memories identified by the `Retriever`.
    *   It intelligently concatenates the instruction, the retrieved memories (sorted chronologically), and the current prompt into a single, coherent `input_ids` sequence.
4.  **Response Generation**:
    *   This new, memory-infused `input_ids` sequence is passed to the base Gemma model's `generate()` method.
    *   Gemma generates a response using the full context provided.
5.  **Memory Storage**:
    *   **After** the response has been generated, the original prompt from the current turn is processed into a summary vector and stored in the memory buffer, ready for future retrieval. The first turn of a session is stored at index `0` and becomes the permanent instruction.

---

## Developer Workflows

### 1. Stage 1 Training: Semantic Alignment

The main training script is `scripts/train_semantic_alignment.py`. It uses a triplet loss function to train **only the `Retriever`'s projection layer**. This teaches the `Retriever` how to map conversational turns to a semantic space where queries and their relevant contexts are close together.

```bash
# This script trains the Retriever and saves the weights.
python3 scripts/train_semantic_alignment.py
```

### 2. Inference and Usage

To use the model with its memory capabilities, run an inference script like `scripts/test_2_turn.py`. The `Lyra` model will automatically load the trained `semantic_retriever.pth` weights upon initialization.

```bash
# Run a multi-turn conversation to test memory retrieval.
python3 scripts/test_2_turn.py
```