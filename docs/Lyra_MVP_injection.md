### **Memory Injection: A Pre-computation Design**

**Context:** The process takes place inside the overridden `generate()` method, immediately after the initial embeddings for the user's prompt have been created.

**Input:** The `generate()` method has access to the `initial_hidden_states` (the embeddings of the user's prompt) and the stateful `self.memory_graph`.

Here is the internal process of the `injection_layer`:

#### **Step 1: Formulate a "Question" (Query Generation)**

The layer first needs to distill the current prompt into a single, focused query to send to the GNN.

1.  **Pooling**: The `generate()` method applies a **mean pooling** operation to the `initial_hidden_states` tensor across the sequence length dimension.
    *   **Input**: `initial_hidden_states` (`[1, seq_len, hidden_dim]`)
    *   **Action**: `torch.mean(initial_hidden_states, dim=1)`
    *   **Output**: A `query_vector` of shape `[1, hidden_dim]`. This vector represents the "gist" of the current prompt.

#### **Step 2: Query the Episodic Memory**

The `query_vector` is now sent to the `EpisodicMemoryGNN` to retrieve relevant memories.

1.  **GNN Forward Pass**: The `injection_layer` calls the GNN's `forward` method with the `query_vector` and the current `memory_graph`.
    *   **Action**: Inside the GNN, a similarity metric (dot-product attention) is computed between the `query_vector` and every node (past episode) in the `memory_graph`.
    *   **Output**: The GNN returns an aggregated **`retrieved_memory`** vector, also of shape `[1, hidden_dim]`. This vector is a blend of the most relevant memories for the current query.
2.  **Handling an Empty Memory**: The GNN is designed to be robust. If the `memory_graph` is empty (e.g., at the start of a conversation), the GNN returns a zero-vector of the correct shape. This ensures the model can function perfectly even without any prior context.

#### **Step 3: Integrate the "Answer" (Context Injection)**

Now the `injection_layer` has the original `initial_hidden_states` and the `retrieved_memory`. It must integrate the memory back into the processing stream.

1.  **Reshaping the Memory**: The `retrieved_memory` is a single vector. It is "unsqueezed" to add a sequence dimension, making its shape `[1, 1, hidden_dim]`.
2.  **Prepending**: The reshaped memory vector is prepended to the `initial_hidden_states` using `torch.cat`.
    *   **Action**: `modified_hidden_states = torch.cat([retrieved_memory, initial_hidden_states], dim=1)`
    *   This creates a new, longer sequence of embeddings that now starts with the retrieved memory.
3.  **Attention Mask Update**: A new `attention_mask` is created that is one token longer than the original, ensuring the model pays attention to the newly injected memory token.

#### **Step 4: Delegate to the `generate` Method**

The `injection_layer`'s job is complete. It outputs the `modified_hidden_states` and `modified_attention_mask`.

*   These tensors are then passed to the `super().generate()` method as `inputs_embeds` and `attention_mask`, respectively. The `transformers` library takes over, using this memory-infused input to generate the response.

This step-by-step design provides a robust and well-defined mechanism for querying and integrating episodic memory, which is fully compatible with the `transformers` generation loop.