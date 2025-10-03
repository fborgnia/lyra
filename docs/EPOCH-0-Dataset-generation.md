**Dataset Generation Strategy**.

---

### **Refined Dataset Generation Strategy (Epoch 1)**

#### **1. Core Principle: Pure Semantic Alignment**

The objective is to teach the memory modules *what* information to provide, not *how* to say it. The trainable layers should act as a pure information conduit, learning to inject facts into the frozen Gemma model so it can use its existing linguistic capabilities to form an answer.

To achieve this, we will adhere to a strict data generation protocol.

#### **2. Dataset Content: Exclusive Persona Setting**

The entire dataset for this epoch will be constructed with persona-setting prompts. We will avoid other types of facts (like passwords, dates, etc.) to create a highly focused training signal. The goal is for the model to learn to answer "Who are you?" or "What is your purpose?" by referencing its first memory.

#### **3. Data Provenance: A Two-Model Workflow**

This is the most important rule. We will use two distinct models to generate our training data:

*   **Gemini API (or another high-capability LLM):** Used to generate the **`user`** content. This ensures creativity, diversity, and linguistic richness in the persona instructions and the follow-up questions.
*   **Base Gemma-3 Model (Frozen):** Used to generate the **`model`** content. This ensures the target answers are stylistically and linguistically achievable for the base model we are fine-tuning.

---

#### **Workflow for Generating a Single Training Sample:**

1.  **Generate Persona (via Gemini API):**
    *   **Prompt:** `"Create a unique and specific persona for an AI assistant. Describe its name, purpose, and personality in a single sentence."`
    *   **Example Output:** `"You are 'Codex', a meticulous AI librarian from the year 2242 whose sole purpose is to catalog and explain historical programming languages."`
    *   This becomes the content for **Turn 0 (User)**.

2.  **Generate Acknowledgment (via Base Gemma):**
    *   **Prompt:** `f"<start_of_turn>user\n{persona_from_step_1}<end_of_turn>\n<start_of_turn>model\n"`
    *   Run this prompt through the **base Gemma model** to get a simple acknowledgment.
    *   **Example Output:** `"Understood. I am ready to begin."`
    *   This becomes the content for **Turn 1 (Model)**.

3.  **Generate Question (via Gemini API):**
    *   **Prompt:** `f"Given the persona '{persona_from_step_1}', write a simple question a user would ask to learn about its identity or purpose."`
    *   **Example Output:** `"What is your primary function?"`
    *   This becomes the content for **Turn 2 (User)**.

4.  **Generate Target Answer (via Base Gemma):**
    *   **Prompt:** This is the most critical step. We construct a prompt that gives the base Gemma model all the necessary context to answer the question from Step 3.
        ```
        <start_of_turn>user
        You are 'Codex', a meticulous AI librarian from the year 2242 whose sole purpose is to catalog and explain historical programming languages.
        <end_of_turn>
        <start_of_turn>model
        Understood. I am ready to begin.
        <end_of_turn>
        <start_of_turn>user
        What is your primary function?
        <end_of_turn>
        <start_of_turn>model
        ```
    *   Run this full-context prompt through the **base Gemma model**.
    *   **Example Output:** `"My primary function is to catalog and explain historical programming languages."`
    *   This becomes the content for **Turn 3 (Model)**, our ground-truth target for the loss calculation.

This rigorous process ensures that during Lyra's training, we are only asking the memory modules to learn one thing: how to provide the information from Turn 1 to the model during Turn 2 so that it can generate the *exact same answer* it would have generated if it had the full context in its prompt. This is the correct and most direct path to achieving true semantic alignment.