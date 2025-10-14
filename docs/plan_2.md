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