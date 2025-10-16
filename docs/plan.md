This is an excellent and ambitious plan. Moving from a sequential block to a parallel, integrated head architecture is a significant step towards a more efficient and powerful model. You've correctly identified all the main challenges: integrating with Grouped-Query Attention (GQA) and correctly handling the two separate contexts (the main growing cache vs. the static loaded cache).

Your proposal to pre-calculate and save the attention mask and positional embeddings along with the KV cache is very insightful. It will save significant computation during inference.

Let's break down how to proceed with a detailed implementation plan.

### Understanding the Core Challenge: Grouped-Query Attention (GQA)

Before we can split the heads, we must understand how Gemma-3 uses them. As you noted, it uses GQA.

*   **Standard Multi-Head Attention (MHA):** Every Query head has its own dedicated Key and Value head (e.g., 16 Q heads, 16 K heads, 16 V heads).
*   **Grouped-Query Attention (GQA):** Multiple Query heads share the same Key and Value head. This saves memory and computation, especially in the KV cache.
    *   For `gemma-3-1b-it`, the config is: `num_attention_heads: 16`, `num_key_value_heads: 8`.
    *   This means `num_key_value_groups` is 2 (16 / 8 = 2). Every 2 Query heads share 1 K/V head.
*   **How it's implemented:** In [`modeling_gemma3.py`]modeling_gemma3.py ), the code projects to the smaller number of K/V heads and then uses `repeat_kv` to expand them to match the number of Query heads right before the attention score calculation.

Our plan to replace half the global heads must respect this grouping. We will replace **8 Q heads** and their corresponding **4 K/V heads**.

---

### Implementation Plan: Parallel Head Integration

Here is a detailed, step-by-step plan to achieve your goal. The most robust way to implement this is to create a new, dedicated `LyraGemma3Attention` class that encapsulates all the complex logic.

#### Step 1: Create the New `LyraGemma3Attention` Class

In [`lyra/lyra.py`]lyra.py ), create a new identical class that replaces `Gemma3Attention`. This will be our main workspace. its functionality is exactly as the original gemma model.

#### Step 2: Modify the `__init__` Method

The normalization layers (`q_norm`, `k_norm`) operate on a per-head basis. Since the head dimension (`head_dim`) is the same for all heads, **we do not need to split the normalization layers themselves.** We can create them once and apply them to both the vanilla and Lyra streams.

1.  **Keep `q_proj` as is:** The original `q_proj` layer remains unchanged.
2.  **Split K/V Projections:** Create separate projection layers for the vanilla and Lyra K/V heads.
    *   `self.vanilla_k_proj`, `self.vanilla_v_proj`
    *   `self.lyra_k_proj`, `self.lyra_v_proj`
3.  **Create Normalization Layers:** Instantiate `q_norm` and `k_norm` exactly as they are in the original `Gemma3Attention`.
    *   `self.q_norm = Gemma3RMSNorm(dim=self.head_dim, ...)`
    *   `self.k_norm = Gemma3RMSNorm(dim=self.head_dim, ...)`
4.  **Load Static Context:** This remains the same. Load the Lyra KV cache, mask, and embeddings as persistent buffers.

#### Step 3: Re-implement the `forward` Method (The Core Logic)

Normalization is applied *after* the linear projection but *before* the rotary position embeddings are applied.
This is where the parallel processing happens. The `forward` pass of your new `LyraGemma3Attention` class will look like this:

1.  **Project Q, K, V:**
    *   Calculate the full query states: `query_states = self.q_proj(hidden_states)`.
    *   Calculate the separate K/V states:
        *   `vanilla_k = self.vanilla_k_proj(hidden_states)`
        *   `lyra_k = self.lyra_k_proj(hidden_states)`
        *   (and the same for `v_proj`)

2.  **Apply Normalization:**
    *   Normalize the full query tensor: `query_states = self.q_norm(query_states)`.
    *   Normalize each of the key tensors separately using the *same* `k_norm` layer:
        *   `vanilla_k = self.k_norm(vanilla_k)`
        *   `lyra_k = self.k_norm(lyra_k)`
    *   (Note: The value states are not normalized in the Gemma architecture).

3.  **Split Q:**
    *   Use `torch.chunk` or slicing to divide the normalized `query_states` into `vanilla_q` and `lyra_q`.

4.  **Process Vanilla Heads (Main Growing Context):**
    *   Apply rotary embeddings to `vanilla_q` and `vanilla_k` using the original `position_embeddings`.
    *   Update the main `past_key_values` cache.
    *   Calculate attention scores and `vanilla_attn_output`.

5.  **Process Lyra Heads (External Static Context):**
    *   Apply the pre-loaded `self.lyra_position_embeddings` to `lyra_q` and `lyra_k`.
    *   Combine the pre-loaded `self.lyra_kv_cache` with the new `lyra_k`/`lyra_v`.
    *   Calculate attention scores and `lyra_attn_output`.

6.  **Combine and Project:**
    *   Concatenate `vanilla_attn_output` and `lyra_attn_output`.
    *   Pass the result through the `o_proj` layer.

#### Step 4: Update the `GemmaInjector`

Your `GemmaInjector` will now be much simpler and cleaner. Instead of replacing the `forward` method of the `Gemma3DecoderLayer`, you will replace the `self_attn` module itself.

```python
# Inside GemmaInjector.enable()
for layer in self.model.model.layers:
    if not layer.self_attn.is_sliding: # Target only global layers
        # Create an instance of our new attention class
        lyra_attn_module = LyraGemma3Attention(self.model.config, layer.layer_idx).to(
            self.model.device, dtype=self.model.dtype
        )
        
        # You will need a custom function to load the weights from the original
        # block into the resized and new projection layers of your Lyra module.
        lyra_attn_module.load_weights_from_original(layer.self_attn)
        
        # Replace the entire attention module in the layer
        layer.self_attn = lyra_attn_module
        print(f"Replaced attention module in layer {layer.layer_idx} with LyraGemma3Attention.")
```

This approach is far more robust and modular. It cleanly isolates all the complex logic within your new `LyraGemma3Attention` class and leaves the `Gemma3DecoderLayer` untouched, which is a much better software engineering practice. This is the path I strongly recommend for your next milestone.