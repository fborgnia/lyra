### **Episodic Memory Storage: A Step-by-Step Design**

**Context:** The model has just received a user prompt and has completed generating its own response. The final token it produced was \<end\_of\_turn\>. This special token acts as the trigger for the entire memory storage procedure.  
**Input:** The model has access to the full context of the just-completed interaction. Specifically, it has the sequence of hidden states from the target intermediate layer (e.g., Layer 10\) for both the user's prompt and its own response.  
Here is the step-by-step internal process that is automatically initiated:

#### **Step 1: Consolidate the Episode's Full Context**

The model first gathers the complete sequence of internal representations that define the episode.

1. **Identify Turn Boundaries**: The model's internal logic uses the \<start\_of\_turn\> and \<end\_of\_turn\> tokens to identify the hidden states corresponding to the user's prompt and the hidden states corresponding to its own response.  
2. **Concatenate Hidden States**: It concatenates these two sequences of hidden states along the sequence dimension.  
   * **Input**: user\_hidden\_states (\[1, user\_seq\_len, hidden\_dim\]) and model\_hidden\_states (\[1, model\_seq\_len, hidden\_dim\]).  
   * **Action**: torch.cat(\[user\_hidden\_states, model\_hidden\_states\], dim=1)  
   * **Output**: A single episode\_hidden\_states tensor of shape \[1, full\_seq\_len, hidden\_dim\]. This tensor is the complete, high-dimensional "recording" of the entire interaction.

#### **Step 2: Create a Coherent Summary (Pooling)**

Now, the model must condense this long sequence of hidden states into a single, fixed-size vector that can represent the node in the GNN.

1. **Apply Pooling Operation**: The model applies a pooling operation across the full\_seq\_len dimension of the episode\_hidden\_states tensor. **Mean pooling** is the standard and effective choice.  
   * **Input**: episode\_hidden\_states (\[1, full\_seq\_len, hidden\_dim\])  
   * **Action**: torch.mean(episode\_hidden\_states, dim=1)  
   * **Output**: A single episode\_vector of shape \[1, hidden\_dim\]. This vector is the semantic summary of the entire user-model interaction.

#### **Step 3: Create the New Node in the Graph**

This episode\_vector is now ready to be permanently stored in the GNN.

1. **Initialize New Node**: A new node is created.  
2. **Assign Features**: The episode\_vector is assigned as the feature vector for this new node. This node now mathematically represents the "memory" of the conversation turn.

#### **Step 4: Store the Node and Update Graph Structure**

Finally, the new node is integrated into the model's internal self.memory\_graph.

1. **Handle First Turn**: If the self.memory\_graph is currently None (i.e., this is the very first turn of the conversation), a new torch\_geometric.data.Data object is created, and this new node is added as the first node.  
2. **Append to Existing Graph**: If the graph already exists, the new node's features are appended to the graph's node feature tensor (graph.x).  
3. **Add Temporal Edge**: To preserve the conversational flow, an edge is added to the graph's edge index (graph.edge\_index). This edge connects the node from the *previous* turn to the node we just created. This chronological link is crucial for allowing the GNN to reason about the order of events.

**Outcome:** The self.memory\_graph is now updated with a new node representing the interaction that just finished. This entire process happens instantly and invisibly. When the user submits their next prompt, the MemoryInjectionLayer will query a graph that is already enriched with the memory of this most recent exchange.