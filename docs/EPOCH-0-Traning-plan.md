### **Training Script Implementation Plan**

This plan creates a custom training script designed specifically for Lyra's two-stage memory process. It will be implemented incrementally to ensure each component is working correctly before integrating the next.

#### **Step 1: The Core Simulation Function**

*   **Goal:** Create a function that performs the two-pass simulation for a single `(U1, U2, M2)` data sample.
*   **Actions:**
    1.  The function will accept the `model`, `tokenizer`, and a data triplet.
    2.  **Pass 1 (Memory Seeding):**
        *   Clear the model's memory store: `model.memory_store.clear()`.
        *   Construct the "past" context (`U1` -> neutral `M1` acknowledgment).
        *   Run a forward pass inside a `torch.no_grad()` block to generate hidden states.
        *   Archive the memory by calling `model.archive_memory()`.
    3.  **Pass 2 (Training):**
        *   Construct the second conversational history (`U2` -> `M2`).
        *   Run a standard forward pass to get the final `logits`.
    4.  Return the `logits` and the tokenized full sequence (`input_ids`).

#### **Step 2: Masked Loss Calculation**

*   **Goal:** Create a function to calculate the loss, focusing only on the target `M2` response.
*   **Actions:**
    1.  The function will accept the `logits` and `input_ids` from Step 1.
    2.  It will create a `labels` tensor by cloning the `input_ids`.
    3.  It will programmatically find the starting token of the `M2` sequence.
    4.  It will set all label tokens *before* the `M2` sequence to `-100`, which is the standard ignore index for PyTorch's loss functions.
    5.  It will compute and return the cross-entropy loss, which will now only apply to the `M2` part of the sequence.

#### **Step 3: The Main Training Loop & Orchestration**

*   **Goal:** Build the main script to orchestrate the entire training process.
*   **Actions:**
    1.  **Setup:**
        *   Load the `Lyra` model and tokenizer.
        *   Identify and freeze all base model parameters (`requires_grad=False`).
        *   Collect only the trainable memory module parameters.
        *   Initialize an optimizer (e.g., `AdamW`) with *only* the trainable parameters.
    2.  **Looping:**
        *   Load the `epoch0_curriculum.jsonl` dataset.
        *   Loop through epochs and then through each sample in the dataset.
        *   In the loop, call the functions from Step 1 and Step 2 to get the `loss`.
        *   Perform the standard backpropagation sequence: `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`.
        *   Add logging to monitor the training loss.

#### **Step 4: Saving the Trained Weights**

*   **Goal:** Persist the results of the training.
*   **Actions:**
    1.  After the training loop completes, create a `state_dict` containing only the trained memory module weights.
    2.  Save this `state_dict` to a file (e.g., `lyra_memory_epoch0.pth`). This small file will contain all the learned knowledge.