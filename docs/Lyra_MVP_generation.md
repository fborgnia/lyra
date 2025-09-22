### **Episodic Memory Storage: A Step-by-Step Design (V2)**

**Context:** The model has just received a user prompt and has completed generating its own response. The final token it produced was `<end_of_turn>`. This special token acts as the trigger for the entire memory storage procedure.

**Input:** The model has access to the hidden states from the target intermediate layer (e.g., Layer 8) for the **user's prompt only**. This is an intentional design choice to create a clean, fact-focused memory, avoiding the noise from the model's conversational responses (e.g., "Okay, I'll remember that.").

Here is the step-by-step internal process that is automatically initiated:

#### **Step 1: Create a Raw Summary (Pooling)**

The model first condenses the sequence of hidden states from the user's prompt into a single, fixed-size vector.

1.  **Apply Pooling Operation**: The model applies a **mean pooling** operation across the sequence length dimension of the prompt's hidden states.
    *   **Input**: `prompt_hidden_states` (`[1, prompt_seq_len, hidden_dim]`)
    *   **Action**: `torch.mean(prompt_hidden_states, dim=1)`
    *   **Output**: A single `raw_episode_vector` of shape `[1, hidden_dim]`. This vector is the initial, unrefined summary of the core information provided by the user.

#### **Step 2: Refine the Summary via GNN**

The `raw_episode_vector` is not stored directly. Instead, it is passed through the **trainable GNN** to be transformed into a semantically rich and aligned representation. This is the core of the "memory encoding" process.

*   **Input**: `raw_episode_vector` (`[1, hidden_dim]`) and the current `memory_graph`.
*   **Action**: `final_episode_vector = gnn(raw_episode_vector, memory_graph)`
*   **Output**: A `final_episode_vector` of shape `[1, hidden_dim]`. This refined vector is now ready to be stored as a memory.

#### **Step 3: Create the New Node in the Graph**

This `final_episode_vector` is now ready to be permanently stored in the GNN.

1.  **Initialize New Node**: A new node is created.
2.  **Assign Features**: The `final_episode_vector` is assigned as the feature vector for this new node. This node now mathematically represents the "memory" of the user's statement.

#### **Step 4: Store the Node and Update Graph Structure**

Finally, the new node is integrated into the model's internal `self.memory_graph`.

1.  **Handle First Turn**: If the `self.memory_graph` is currently `None` (i.e., this is the very first turn of the conversation), a new `torch_geometric.data.Data` object is created, and this new node is added as the first node.
2.  **Append to Existing Graph**: If the graph already exists, the new node's features are appended to the graph's node feature tensor (`graph.x`).
3.  **Add Temporal Edge**: To preserve the conversational flow, an edge is added to the graph's edge index (`graph.edge_index`). This edge connects the node from the *previous* turn to the node we just created. This chronological link is crucial for allowing the GNN to reason about the order of events.

**Outcome:** The `self.memory_graph` is now updated with a new, refined node representing the core information from the user's last turn. This entire process happens instantly and invisibly, preparing the model for the next interaction.