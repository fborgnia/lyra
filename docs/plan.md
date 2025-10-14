Here is a step-by-step plan to implement the Gemma 3 plugin.

### **Phase 1: Foundational Implementation & Initial Test**

This phase focuses on creating a functional plugin that injects a sequential attention block using the model's existing KV cache.

**Step 1: Project Scaffolding and Code Duplication**
1.  Create a new directory named lyra inside your project root Lyra.
2.  Create a new file `lyra/gemma_components.py`. This file will house the necessary Gemma 3 classes you'll need to modify or instantiate independently.
3.  From modeling_gemma3.py, copy the following class definitions into `lyra/gemma_components.py`:
    *   `Gemma3RMSNorm`
    *   `Gemma3Attention`
    *   `Gemma3MLP` (if you later decide to inject a full block, but for now `Gemma3Attention` is the priority)
4.  Create the main plugin file: `lyra/lyra.py`.

**Step 2: Create the Plugin's Attention Block**
1.  In `lyra/lyra.py`, define a new class `InjectedAttentionBlock` that inherits from `torch.nn.Module`.
2.  The `__init__` method of this class will instantiate a `Gemma3Attention` block (from your `lyra/gemma_components.py` file). It should accept a Gemma 3 configuration object and a layer index to properly initialize the attention block.
3.  The `forward` method of `InjectedAttentionBlock` will simply call the forward method of its internal `Gemma3Attention` instance, passing through all required arguments like `hidden_states`, `position_embeddings`, `attention_mask`, and `past_key_values`.

**Step 3: Develop the Core Plugin Logic**
1.  In `lyra/lyra.py`, create the main plugin class, for example, `GemmaInjector`.
2.  The `__init__` method of `GemmaInjector` will accept a pre-loaded Gemma 3 model instance as an argument.
3.  Implement a method within `GemmaInjector` called `attach()`. This method will perform the model modification (monkey-patching).
4.  Inside `attach()`, iterate through the layers of the passed-in model (e.g., `model.model.layers`).
5.  For each layer, check if it contains a global attention block. You can determine this by inspecting the layer's configuration or a property, like `layer.self_attn.is_sliding` being `False`.
6.  When a global attention layer is found, instantiate your `InjectedAttentionBlock`.
7.  Initialize the weights of your new block by deep-copying the weights from the original global attention block in that same layer (`layer.self_attn`). This ensures it starts with the same pre-trained parameters.
8.  Store the new `InjectedAttentionBlock` instance as a new attribute on the model's layer object (e.g., `layer.injected_attn_block`).
9.  Modify the `forward` method of the identified `Gemma3DecoderLayer` to include a call to your injected block. The new execution flow within the layer should be: original `self_attn` -> `post_attention_layernorm` -> `residual_add` -> **your `injected_attn_block`**. The output of your block will then feed into the next part of the original layer's logic (the `pre_feedforward_layernorm`).

**Step 4: First Testable Version**
1.  At this stage, your injected block is using the same `past_key_values` cache as the original attention mechanism because you are passing it through in the modified `forward` method.
2.  Develop a simple inference script to load the Gemma 3 1B IT model, apply your `GemmaInjector` plugin, and run a text generation task.
3.  The goal is to verify that the model runs without crashing and to establish a baseline for performance and output quality, which you can use to measure the impact of the sequential injection.

---

### **Phase 2: Implementing the Separate KV Cache**

This phase modifies the plugin to use an independent context via a separate KV cache.

**Step 5: Modify the Injected Block for a New Cache**
1.  Update the `InjectedAttentionBlock` class in `lyra/lyra.py`.
2.  Modify its `forward` method to accept an additional argument: `external_past_key_values`.
3.  Inside this `forward` method, ensure that the call to the underlying `Gemma3Attention` block uses this new `external_past_key_values` instead of the model's default cache.

**Step 6: Update the Plugin to Manage the External Cache**
1.  Modify the `GemmaInjector`'s `__init__` or `attach` method to accept a new argument: `external_kv_cache`. This will be the separate, pre-filled cache object you intend to provide.
2.  Store this external cache within the `GemmaInjector` instance.
3.  Update the modified `forward` method of the `Gemma3DecoderLayer`. The call to your `injected_attn_block` must now pass this `external_kv_cache`.

**Step 7: Prepare and Inject the External Cache**
1.  You will need a mechanism to create and populate the `external_kv_cache`. This involves running a separate context through the model (or a compatible model) and capturing the resulting `past_key_values`.
2.  In your main script, after creating this cache, you will pass it to the `GemmaInjector` during its initialization or attachment phase.

**Step 8: Final Implementation and Verification**
1.  With these changes, the injected attention block now operates on the main hidden states but uses its own separate key-value context for the attention calculation.
2.  Run your inference script again. The model should now generate outputs that are influenced by both the primary input and the context provided in the external KV cache.
3.  At this point, the implementation is complete and ready for the fine-tuning work you will conduct separately.