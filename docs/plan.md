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

In [`lyra/lyra.py`]lyra.py ), create a new class that inherits from `Gemma3Attention`. This will be our main workspace.

#### Step 2: Modify the `__init__` Method

The constructor needs to be aware of the head split and load the static Lyra context.

1.  **Resize Original Projections:** The original `q_proj`, `k_proj`, and `v_proj` will now only handle half the heads (the "vanilla" global heads). You will need to resize their weight matrices.
2.  **Create Lyra Projections:** Add new `lyra_k_proj` and `lyra_v_proj` layers. We can reuse the main `q_proj` for all queries and split the resulting tensor, since the query always comes from the same `hidden_states`.
3.  **Load and Store Static Context:** In the `__init__` method, load your serialized Lyra context (KV cache, attention mask, and positional embeddings) from files and store them as persistent buffers on the module. This ensures they are moved to the correct device with the model and are only loaded once.

#### Step 3: Re-implement the `forward` Method (The Core Logic)

This is where the parallel processing happens. The `forward` pass of your new `LyraGemma3Attention` class will look like this:

1.  **Project Q, K, V:**
    *   Calculate the full query states for all heads: `query_states = self.q_proj(hidden_states)`.
    *   Split the `query_states` tensor along the head dimension into `vanilla_q` and `lyra_q`.
    *   Calculate the vanilla K/V states for the main context: `vanilla_k = self.k_proj(hidden_states)` and `vanilla_v = self.v_proj(hidden_states)`.
    *   Calculate the Lyra K/V states. **Crucially, these are also projected from the same input `hidden_states`**: `lyra_k = self.lyra_k_proj(hidden_states)` and `lyra_v = self.lyra_v_proj(hidden_states)`.

2.  **Process Vanilla Heads (Main Growing Context):**
    *   Apply rotary embeddings to `vanilla_q` and `vanilla_k` using the original `position_embeddings` passed into the function.
    *   Update the main `past_key_values` cache with the new `vanilla_k` and `vanilla_v`.
    *   Use `repeat_kv` on the vanilla K/V heads from the cache.
    *   Calculate attention scores using `vanilla_q`, the repeated vanilla K/V, and the original `attention_mask`.
    *   Compute the `vanilla_attn_output`.

3.  **Process Lyra Heads (External Static Context):**
    *   **Apply Static Embeddings:** Apply the pre-loaded `self.lyra_position_embeddings` to `lyra_q` and the newly computed `lyra_k`.
    *   **Combine Caches:** The final Lyra key/value states for this pass are a combination of the pre-loaded `self.lyra_kv_cache` and the newly computed `lyra_k`/`lyra_v` for the current token.
    *   Use `repeat_kv` on the combined Lyra K/V heads.
    *   Calculate attention scores using `lyra_q`, the repeated Lyra K/V, and the pre-loaded `self.lyra_attention_mask`.
    *   Compute the `lyra_attn_output`.

4.  **Combine and Project:**
    *   Concatenate the `vanilla_attn_output` and `lyra_attn_output` back into a single tensor along the head dimension.
    *   Pass this combined tensor through the original `o_proj` layer to get the final output.

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