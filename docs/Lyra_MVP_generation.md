### **Episodic Memory Storage: A Pre-computation Design**

**Context:** The model has just received a user prompt via a call to its `generate()` method. The memory storage process is the *first* thing that happens, before the model begins generating its response.

**Input:** The model has access to the `input_ids` for the user's current prompt.

Here is the step-by-step internal process that is automatically initiated within the overridden `generate()` method:

#### **Step 1: Create Initial Embeddings**

The model first converts the user's tokenized prompt (`input_ids`) into its initial embedding representation.

*   **Action**: `initial_hidden_states = self.model.embed_tokens(input_ids)`
*   **Output**: A tensor of shape `[1, prompt_seq_len, hidden_dim]`.

#### **Step 2: Create a Raw Summary (Pooling)**

The model then condenses this sequence of embeddings into a single, fixed-size vector that represents the "gist" of the prompt. This happens inside the `_update_memory` helper method.

1.  **Apply Pooling Operation**: The model applies a **mean pooling** operation across the sequence length dimension of the prompt's hidden states.
    *   **Input**: `initial_hidden_states` (`[1, prompt_seq_len, hidden_dim]`)
    *   **Action**: `torch.mean(initial_hidden_states, dim=1)`
    *   **Output**: A single `turn_summary_vector` of shape `[1, hidden_dim]`. This vector is the raw summary of the information provided by the user.

#### **Step 3: Store the Node and Update Graph Structure**

This `turn_summary_vector` is now ready to be permanently stored in the model's stateful `self.memory_graph`.

1.  **Append to Existing Graph**: The new `turn_summary_vector` is appended to the graph's node feature tensor (`self.memory_graph.x`).
2.  **Add Temporal Edge**: To preserve the conversational flow, an edge is added to the graph's edge index (`self.memory_graph.edge_index`). This edge connects the node from the *previous* turn to the node we just created. This chronological link is crucial for allowing the GNN to reason about the order of events.

**Outcome:** The `self.memory_graph` is now updated with a new node representing the core information from the user's current prompt. This entire process happens instantly and invisibly, right before the model proceeds with memory injection and response generation.