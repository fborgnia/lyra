# Lyra: A Gemma-based LLM with In-Layer Conversational State Injection

Lyra is an experimental stateful architecture that explores a pattern for conversational context management in Transformer-based models. Built upon a standard Gemma instruction-tuned model, Lyra introduces a system for injecting turn-level state directly into the residual stream of each decoder layer.

This approach uses trainable cross-attention modules to process a composite memory block representing all previous conversational turns. This provides historical context at every stage of processing. The design aims to offer an alternative to managing long-term context through ever-growing prompts. The implementation follows a Parameter-Efficient Fine-Tuning (PEFT) strategy, where the base model's weights remain frozen, and only the lightweight state injection components are trained.

# Experimental Plan: Testing the Core Hypothesis

The central hypothesis is that **context management through a dedicated, in-layer cross-attention block is a viable and effective mechanism.** The experiment is designed to test if this injection system can learn to successfully leverage historical context to improve conversational responses.

### 1. Operationalizing the Hypothesis

The hypothesis will be tested by evaluating the model's ability to maintain context and answer questions that require information from previous turns in a conversation. Success is defined as the model demonstrating a measurable improvement in context-dependent tasks compared to a baseline without the memory injection mechanism.

### 2. Experimental Procedure

The experiment will proceed in two main stages:

**Stage A: Model Fine-Tuning**
The model will be fine-tuned on a general multi-turn conversational dataset. This stage ensures the injection modules are functional and learn to produce meaningful contextual enrichments that the base model can utilize.

**Stage B: Evaluation**
Using a held-out test set of multi-turn dialogues, we will evaluate the model's performance. The test set will include dialogues that specifically require recalling information from earlier turns.

### 3. Metrics and Validation

We will use two primary methods to validate the hypothesis:

*   **Task Performance:**
    *   **Metric:** We will measure standard language modeling metrics (e.g., perplexity) on the held-out conversational dataset. We will also evaluate performance on a question-answering task where the answer is only present in a prior turn.
    *   **Validation:** A successful outcome will show a significant improvement in these metrics over a baseline model where the memory injection modules are disabled.

*   **Attention Weight Analysis:**
    *   **Metric:** For each `MemoryInjectionBlock`, we will record the cross-attention weights applied to the composite memory block.
    *   **Validation:** We expect to see meaningful, non-uniform attention patterns. For queries that require specific historical information, we should observe attention weights focusing on the parts of the memory block corresponding to the relevant prior turn. This would provide qualitative evidence that the mechanism is functioning as intended.

A successful outcome, where the model demonstrates improved performance on context-dependent tasks, would provide strong evidence for the viability of this architectural pattern.

## Architecture Overview: How It Works

The core of Lyra's architecture is the in-place modification of Gemma's decoder layers. During model initialization, each `GemmaDecoderLayer` is augmented with a `MemoryInjectionBlock`, turning it into a `LyraDecoderLayer`. This is achieved by dynamically replacing the `forward` method of each layer.

### Key Components:

-   **`lyra/model.py` (`Lyra`)**: The main model class that subclasses `GemmaForCausalLM`. It orchestrates the dynamic method replacement process and manages the `ConversationalStateStore`. It also overrides the `generate()` method to manage the archival of conversational turns.
-   **`lyra/memory/store.py` (`ConversationalStateStore`)**: A session-level cache that stores the `hidden_states` and `attention_mask` of past conversational turns. Before each generation pass, it **composes all stored turns into a single composite memory tensor**.
-   **`lyra/memory/layer.py` (`LyraDecoderLayer`)**: Not a class that is instantiated, but a new `forward` method that is injected into each of Gemma's existing decoder layers. It contains the logic for integrating historical state into the data path.
-   **`lyra/memory/injection.py` (`MemoryInjectionBlock`)**: The heart of the state injection system. An instance of this block is created for **each decoder layer**. It is responsible for receiving the composite memory block, projecting the current hidden state into a query, and performing cross-attention.
-   **`lyra/memory/attention.py` (`MemoryCrossAttention`)**: A trainable cross-attention module that calculates attention scores between the current layer's `query_states` and the composite `hidden_states` of all past conversational turns.

---

## The Data Flow: A Step-by-Step Guide

When `model.generate()` is called, the following happens at **each decoder layer**:

1.  **Normalization**: The layer's input `hidden_states` are passed through the standard `input_layernorm`.
2.  **State Injection Branch Activation**:
    *   A residual connection (`memory_residual`) is created from the normalized `hidden_states`.
    *   The `MemoryInjectionBlock` for that specific layer is called.
3.  **State Injection**:
    *   The block receives the **composite memory tensor**, which contains the hidden states of all previous turns, from the `ConversationalStateStore`.
    *   It uses its unique, layer-specific `q_proj` layer to project the current `hidden_states` into `query_states`.
    *   It calls the `MemoryCrossAttention` module, which attends to the composite past state using the new `query_states`.
    *   The output is an `aggregated_memory_enrichment` tensor.
4.  **Residual Connection**:
    *   The `memory_output` is passed through its own `post_memory_layernorm`.
    *   The normalized output is added back to the `memory_residual`.
5.  **Self-Attention**: This new, state-infused `hidden_states` tensor is then passed to the original, **frozen** `self_attn` block of the Gemma layer.
6.  **Archival**: After the full model response is generated, the `Lyra` model takes the final `hidden_states` of the prompt-response pair and stores them in the `ConversationalStateStore` for use in subsequent turns.

---

## Developer Workflows

### 1. Stage 1 Training: Fine-Tuning the State Injection Modules

The training script `scripts/train_memory_heads.py` is used to fine-tune the model. This script employs a PEFT strategy where all of the original Gemma model's parameters are frozen.

Only the newly added components are trained:
- The `q_proj` in each `MemoryInjectionBlock`.
- The `k_proj`, `v_proj`, and `o_proj` in the `MemoryCrossAttention` module.
- The `post_memory_layernorm` in each decoder layer.

This process teaches the injection system how to generate outputs that the frozen self-attention and FFN layers can interpret, effectively learning to "speak Gemma's language."

```bash
# This script will train the new state injection modules.
python3 scripts/train_memory_heads.py
```

### 2. Inference and Usage

To use the model, run an inference script like `scripts/test_cmd_1_turn.py`. The `Lyra` model will use its trained modules to provide context-aware responses across multiple turns.

```bash
# Run a multi-turn conversation to test the architecture.
python3 scripts/test_cmd_1_turn.py
```