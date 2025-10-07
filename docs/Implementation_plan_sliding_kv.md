### **Project Plan: Sliding Window KV Cache Implementation**

**Phase 1: Code Refactoring and Baseline Establishment**

1.  **Strip Down the `Lyra` Class:** Your current model.py is built around the state-injection hypothesis. This must be dismantled. Refactor the `Lyra` class to be a minimal subclass of `Gemma3ForCausalLM`. Remove all logic related to `MemoryInjectionBlock`, `EpisodicMemoryStore`, dynamic method replacement, and the custom `generate` method override. The goal is a clean class that simply loads the pre-trained Gemma model.

2.  **Archive the State-Injection Module:** The code in memory is now deprecated. Move the entire memory directory to `lyra/archive/injection_v1/` to preserve the prior art without cluttering the current workspace.

3.  **Create the Custom Attention Module Stub:** Create a new file: `lyra/attention.py`. In this file, define a new class, `LyraAttention`, which will inherit from `torch.nn.Module`. For now, this class will be an empty stub.

4.  **Replace Gemma's Attention with a Custom Module:** Modify the `Lyra` class's `__init__` method. After loading the pre-trained model, iterate through each decoder layer and replace its `self_attn` module with an instance of your new `LyraAttention` class. This establishes the surgical control we need over the attention mechanism.

**Phase 2: Replicating Gemma's Native Attention**

5.  **Implement a 1:1 Copy of `Gemma3Attention`:** Populate the `LyraAttention` class in `lyra/attention.py`. You must replicate the functionality of the `Gemma3Attention` class from the `transformers` library precisely. This includes the Q, K, V projections, rotary position embeddings (`RoPE`), and output projection. Copy the pre-trained weights from the original `self_attn` module into your new `LyraAttention` instance for each layer.

6.  **Verification Pass:** This step is non-negotiable. You must prove that your `LyraAttention` module is a perfect functional replacement. Perform a forward pass on the model with a fixed input and verify that the output logits are bit-for-bit identical to the output of the original, unmodified Gemma model. Any deviation indicates an error in your replication.

**Phase 3: Implementing the Sliding Window KV Cache**

7.  **Introduce the Sliding Window Cache Logic:** Modify the `LyraAttention.forward` method. The standard implementation accepts an optional `past_key_value` tuple. You will augment this logic. Introduce a `window_size` parameter. When the sequence length of the keys and values in the cache exceeds `window_size`, you will slice the K and V tensors to discard the oldest tokens beyond the window.

8.  **Manage Rotary Embedding Positions:** Crucially, your rotary embeddings must account for the sliding window. The `position_ids` passed to the `RoPE` module must reflect the true position of each token in the sequence, not its index within the truncated cache. You will need to manage and pass the correct `position_ids` during the forward pass, ensuring they increment correctly even as the cache slides.
