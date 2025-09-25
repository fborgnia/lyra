# Lyra: A Gemma-based Language Model with Episodic Memory

Lyra is an experimental language model that enhances a standard Gemma instruction-tuned model with an internal episodic memory. This allows the model to maintain a persistent, structured memory of conversations, enabling it to recall past information and provide more context-aware responses.

## Architecture Overview: How It Works

The core of the project is the `Lyra` model, which subclasses `transformers.Gemma3ForCausalLM`. It integrates a small, trainable `Retriever` module that acts as a semantic search engine over the conversational history.

The system's key innovation is its **text-based memory injection**. Instead of injecting abstract memory vectors, Lyra reconstructs a coherent, multi-turn chat history that the base Gemma model can natively understand.

### Key Components:

-   **`lyra/model.py` (`Lyra`)**: The main model class. It orchestrates the entire memory process, overriding the `generate()` method to intercept user prompts and manage the memory workflow.
-   **`lyra/retriever.py` (`Retriever`)**: A lightweight, trainable module. Its sole purpose is to find the most semantically relevant memory from the past. It takes a query vector (from the current prompt) and returns the **index** of the best-matching memory.
-   **`lyra/injection.py` (`MemoryInjectionLayer`)**: The "prompt engineer." It takes the index provided by the `Retriever`, retrieves the corresponding past conversation turn, and stitches it together with the current prompt to create a single, valid, multi-turn input for the base model.
-   **The Memory Buffer**: Implemented as a simple Python `list` within the `Lyra` class. Each item is a dictionary containing the semantic summary vector and the original `input_ids` of a conversational turn.

---

## The Inference Loop: A Step-by-Step Guide

When a user calls `model.generate()`:

1.  **Query**: The user's prompt is received.
2.  **Memory Retrieval (The Retriever's Job)**:
    *   The `Retriever` compares the semantic vector of the current prompt against the vectors of all past turns stored in the memory buffer.
    *   It identifies the best match and returns its index (e.g., `0` for the first turn).
3.  **Prompt Engineering (The Injection Layer's Job)**:
    *   The `MemoryInjectionLayer` retrieves the stored `input_ids` for the chosen memory index.
    *   It intelligently concatenates the retrieved `input_ids` (the past turn) with the `input_ids` of the current prompt, ensuring only one `<bos>` token is present.
    *   The result is a single, coherent `input_ids` sequence that looks like a natural multi-turn conversation.
4.  **Response Generation**:
    *   This new, memory-infused `input_ids` sequence is passed to the base Gemma model's `generate()` method.
    *   Gemma generates a response using the full context provided.
5.  **Memory Storage**:
    *   **After** the response has been generated, the original prompt from the current turn is processed into a summary vector and stored in the memory buffer, ready for future retrieval.

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

