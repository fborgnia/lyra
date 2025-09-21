### **Final MVP Plan: Gemma Instruct with Fully Internalized Episodic Memory**

This document outlines the implementation of a GemmaWithMemory model built upon a **Gemma instruction-tuned base**. The model will be **fully compatible** with the standard Gemma Instruct interface by using its specialized turn tokens to implicitly manage an internal episodic memory graph.

#### **Phase 1: Foundation & "Instruct-Aware" Stateful Model**

**Goal**: Build the model class with internalized state management logic that is tightly coupled with the Gemma Instruct conversational structure.

1. **Project Setup**:  
   * Standard project setup with transformers, torch, and torch\_geometric.  
2. **Internalized GemmaWithMemory Model (model.py)**:  
   * The \_\_init\_\_ method will:  
     * Load a frozen **Gemma instruction-tuned model** and its corresponding tokenizer.  
     * Store the token IDs for \<start\_of\_turn\> and \<end\_of\_turn\> for quick lookup.  
     * Initialize the GNN, Injection Layer, and the internal memory state: self.memory\_graph \= None.  
   * **No public memory control methods** (e.g., update\_memory, clear\_memory) will be exposed.  
   * Implement an **"Instruct-Aware" forward method**:  
     * At the beginning of the call, it will check if the token ID for \<start\_of\_turn\> is present in the input.  
     * If it is, the model triggers an **internal memory reset**, preparing for a new conversation.  
   * Implement a **custom generate method**:  
     * This method will wrap the standard generate function.  
     * **After** a complete turn is generated (ending with the \<end\_of\_turn\> token), the method will trigger an **internal memory update**. It will automatically summarize the preceding turn and add it as a new node to self.memory\_graph.

#### **Phase 2: Curriculum Stage 1 \- Semantic Alignment**

**(This phase remains unchanged)**. The goal is to train the summarizer head using a triplet loss on individual conversational turns. This ensures the model learns to create high-quality, semantically meaningful node embeddings, which is a prerequisite for effective memory retrieval.

#### **Phase 3: Curriculum Stage 2 \- Retrieval Skill Training**

**Goal**: Fine-tune the model to use its implicit memory management during a standard conversational fine-tuning process.

1. **Data Preparation**:  
   * Use a standard multi-turn conversational dataset.  
   * The critical step is to format every dialogue in the dataset using the **exact Gemma Instruct template**.  
   * Example format: \<start\_of\_turn\>user\\n{user\_prompt}\<end\_of\_turn\>\\n\<start\_of\_turn\>model\\n{model\_response}\<end\_of\_turn\>  
2. **Build the Training Script**:  
   * The training script is now a **standard conversational fine-tuning script**.  
   * It loads the model from Phase 1 and the specially formatted data.  
   * The training loop is a simple, stateless for batch in dataloader:.  
   * The model's internal logic, triggered by the \<start\_of\_turn\> and \<end\_of\_turn\> tokens present in the data, will handle all memory management automatically and transparently. The model learns to use the memory that it is implicitly building to improve its next-token prediction loss.

#### **Phase 4: Evaluation and Iteration**

**Goal**: Verify that the model can be used exactly like a standard Gemma Instruct model while demonstrating long-term memory.

1. **Build an Inference Script (chat.py)**:  
   * The inference script is identical to one you would write for a standard Gemma Instruct model.  
   * It loads the model and tokenizer using AutoModelForCausalLM.from\_pretrained(...).  
   * It uses the standard chat template to format user input.  
   * It calls model.generate(...) to get the response.  
   * **There are absolutely no calls to any special memory methods.** The user interacts with the model as if it were completely normal, but in the background, the model is building and querying its memory graph, leading to more coherent and context-aware conversations.