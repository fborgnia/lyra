### **\#\# Phase 1: Foundation & -   The `forward` method signature must be compatible with the base Gemma model to work with `generate()`. It will *not* accept an external `memory_graph` argument, instead always relying on its internal state.
    -   **During training and inference**, the model will use its internal `self.memory_graph`.

    **Conceptual Code:**
    ```python
    # Inside the GemmaWithMemory forward method
    def forward(self, input_ids, attention_mask=None, ...): # No more memory_graph argument
        # Always use the internal graph
        active_memory_graph = self.memory_graphScaffolding (Time: \~1 Week)**

**Goal:** Set up the project environment and create the individual, standalone modules. The focus here is on getting each component working in isolation before trying to connect them.

1.  **Project Setup:**
    *   Initialize a Git repository for version control.
    *   Create a Python virtual environment (using conda or venv).
    *   Install the core libraries: transformers, torch, and torch\_geometric.
2.  **Frozen Gemma Loader:**
    *   Create a Python script (gemma\_loader.py) that handles loading the Gemma 1B model from Hugging Face.
    *   Implement a critical function that iterates through all of Gemma's parameters and sets param.requires\_grad \= False. This ensures the massive base model remains frozen.
3.  **Standalone GNN Module (gnn.py):**
    *   Define the EpisodicMemoryGNN class using PyTorch Geometric. Start with a simple architecture, like one or two GATConv layers, which are well-suited for learning relationships between nodes.
    *   Write a test script to verify this module independently. You should be able to create an instance, add a few dummy nodes and edges to a torch\_geometric.data.Data object, and perform a forward pass with a random query vector.
4.  **Standalone Memory Injection Layer (injection.py):**
    *   Define the MemoryInjectionLayer class. Its \_\_init\_\_ will accept a GNN model as an argument.
    *   Implement the simple forward method: pool the input hidden states to get a query, pass it to the GNN, get the context back, project it to the correct dimension, and add it to the input hidden states.
    *   Write a test for this layer using a mock GNN to ensure the tensor shapes are correct throughout the process.
5.  **Synthetic Dataset Generator (generate\_dataset.py):**
    *   Create a script to programmatically generate your training data.
    *   It should produce a structured file (like JSON or CSV) where each entry contains a list of context facts, a memory-dependent question, and the target answer. Include the negative examples (questions whose answers are not in the context) as well.

### **\#\# Phase 2: Full Model Integration (Time: \~1 Week)**

**Goal:** Assemble the standalone components into a single, cohesive, and Hugging Face-compatible model architecture.

1.  **Create the Main Model Class (`model.py`):**
    *   This class, `GemmaWithMemory`, will inherit from `GemmaPreTrainedModel` to ensure compatibility. Its `__init__` method will:
        *   Load the frozen Gemma model using your loader from Phase 1.
        *   Instantiate your `EpisodicMemoryGNN`.
        *   Instantiate your `MemoryInjectionLayer`, passing the GNN instance to it.
        *   Initialize an internal state for the memory graph: `self.memory_graph = None`.
2.  **Implement the Custom Forward Pass:**
    *   The `forward` method signature must be compatible with the base Gemma model to work with `generate()`. It will accept an optional `memory_graph` argument.
    *   **During training**, this `memory_graph` will be provided by the data loader.
    *   **During inference** (when using `generate()`), this argument will be `None`, and the model will use its internal `self.memory_graph`.

    **Conceptual Code:**
    ```python
    # Inside the GemmaWithMemory forward method
    def forward(self, input_ids, attention_mask=None, memory_graph=None, ...):
        # Use internal graph if no graph is passed (for HF .generate() compatibility)
        active_memory_graph = memory_graph if memory_graph is not None else self.memory_graph

        hidden_states = self.gemma.model.embed_tokens(input_ids)

        # First part of Gemma
        for i in range(8):
            hidden_states = self.gemma.model.layers[i](hidden_states)[0]

        # Your custom logic!
        if active_memory_graph is not None:
            hidden_states = self.injection_layer(hidden_states, active_memory_graph)

        # Remainder of Gemma
        for i in range(8, len(self.gemma.model.layers)):
            hidden_states = self.gemma.model.layers[i](hidden_states)[0]

        logits = self.gemma.lm_head(hidden_states)
        return CausalLMOutputWithPast(logits=logits, ...)
    ```

3.  **Implement Public Memory Methods:**
    *   Add methods to the `GemmaWithMemory` class to manage the internal state from an external application:
        *   `add_to_memory(self, text: str)`: Converts text to an embedding and adds it as a node to `self.memory_graph`.
        *   `clear_memory(self)`: Resets `self.memory_graph` to `None`.

### **\#\# Phase 3: The Stateful Training Pipeline (Time: \~2 Weeks)**

**Goal:** Build the specialized data loading and training loop required to teach your stateful model.

1.  **Custom Data Loader:**
    *   Implement a PyTorch `Dataset` class that reads the synthetic data file.
    *   The `collate_fn` for the `DataLoader` should be simple. It will batch the raw context sentences, questions, and answers. It no longer needs to construct the GNN graph.
2.  **The Training Script (train.py):**
    *   Initialize your `GemmaWithMemory` model.
    *   Initialize an optimizer (e.g., AdamW). **Crucially, pass only the trainable parameters to it**: `optimizer = AdamW(list(model.gnn.parameters()) + list(model.injection_layer.parameters()))`.
    *   Write the training loop. For each item in the batch, it will:
        1.  Call `model.clear_memory()`.
        2.  Loop through the context sentences, calling `model.add_to_memory()` for each one.
        3.  Pass the tokenized question to the model's `forward` method.
    *   Compute `CrossEntropyLoss` and perform backpropagation.
3.  **Verification Run:**
    *   Run the training script on a small fraction of your data to confirm that the loss is steadily decreasing. This proves that gradients are flowing correctly.

### **\#\# Phase 4: Packaging and Usage Example (Time: ~3 Days)**

**Goal:** Finalize the model into a reusable, stateful class and create a script demonstrating its use.

1.  **Finalize `GemmaWithMemory` Class:**
    *   Ensure the class correctly inherits from the appropriate Hugging Face `PreTrainedModel` class.
    *   Confirm the `forward` signature and internal state management work as designed.
    *   Make sure the model can be saved using `model.save_pretrained('./trained_model')` and reloaded using `GemmaWithMemory.from_pretrained('./trained_model')`.
2.  **Create a Usage Example Script (`usage_example.py`):**
    *   This script will demonstrate the intended workflow for your model.
    *   It will:
        1.  Load the fine-tuned model and tokenizer from the saved directory.
        2.  Call `model.clear_memory()` to start with a clean state.
        3.  Add context sentences to the model's internal memory: `model.add_to_memory("Fact 1...")`, `model.add_to_memory("Fact 2...")`.
        4.  Tokenize a question (e.g., "What is fact 2?").
        5.  Call the standard `model.generate()` method to get the answer, demonstrating compatibility.
        6.  Print the decoded result.
3.  **Iterate and Refine:**
    *   Based on the results from the usage example, you can now iterate on the model. Is the GNN complex enough? Is the injection layer at the right depth? Is the synthetic dataset diverse enough? This phase involves cycles of analysis, tweaking, and re-training.