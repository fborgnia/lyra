import torch
import torch.nn as nn
import copy
from typing import Optional
from transformers.cache_utils import Cache
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer, Gemma3Attention

class CrossAttentionBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        # This is the injected attention module
        self.self_attn = Gemma3Attention(config, layer_idx)

    def forward(self, *args, **kwargs):
        # Pass all arguments through to the underlying attention module
        return self.self_attn(*args, **kwargs)

class GemmaInjector:
    def __init__(self, model):
        self.model = model

    def enable(self):
        # Define the new forward method once, outside the loop
        def modified_gemma_decoder_layer_forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings_global: torch.Tensor,
            position_embeddings_local: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = True,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
        ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
            
            # --- Lyra Modification: Create a snapshot of the cache ---
            # This copy represents the cache state *before* this layer's self-attention runs.
            # It correctly captures the history from previous tokens and layers.
            kv_cache_copy = copy.deepcopy(past_key_values) if past_key_values is not None else None
            
            # 1. Original Self-Attention Block
            # This block will use and update the main `past_key_values` object in-place.
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)

            position_embeddings = position_embeddings_local if self.self_attn.is_sliding else position_embeddings_global

            # The original forward method returns a tuple, where the last element might be the cache.
            # We need to capture all outputs to correctly reconstruct the final return value.
            attn_outputs = self.self_attn(
                hidden_states=hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            
            hidden_states = attn_outputs[0]
            self_attn_weights = attn_outputs[1] if output_attentions else None

            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + hidden_states

            # 2. ---- Injected Lyra Cross-Attention Block ----
            if hasattr(self, 'cross_attn_block'):
                #kv_cache_copy = torch.load('/home/fede/Projects/Lyra/data/test_kv_cache.pth', weights_only=False)
                #print(f"[Layer {self.layer_idx}] Injecting Lyra Cross-Attention Block", flush=True)
                residual = hidden_states
                # The Lyra block uses the copied cache, which has the correct dimensions
                # for the attention mask and avoids corrupting the main cache.
                cross_attn_hidden_states, _, *_ = self.cross_attn_block(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=kv_cache_copy, # Use the copied cache
                    position_ids=position_ids,
                    output_attentions=False, # We don't need the weights from this block
                    use_cache=use_cache,
                    cache_position=cache_position,
                    **kwargs,
                )
                hidden_states = self.post_attention_layernorm(cross_attn_hidden_states)
                hidden_states = residual + hidden_states
            # ---- End Injected Block ----
            
            # 3. Original MLP (Feed-Forward) Block
            #print(f"[Layer {self.layer_idx}] Cache Length Before MLP: {past_key_values[0][0].shape[2] if past_key_values else 0}", flush=True)
            residual = hidden_states
            hidden_states = self.pre_feedforward_layernorm(hidden_states)
            hidden_states = self.mlp(hidden_states)
            hidden_states = self.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states

            # Reconstruct the output tuple exactly as the original `Gemma3DecoderLayer` does.
            outputs = (hidden_states,)
            if output_attentions:
                outputs += (self_attn_weights,)
            
            # The original `Gemma3DecoderLayer` does NOT return the cache.
            # The cache object is updated in-place and the top-level model handles returning it.
            # We must replicate this behavior.

            return outputs

        # Iterate through the model layers to apply the patch
        for layer in self.model.model.layers:
            # Inject block only in global attention layers
            if not layer.self_attn.is_sliding:
                # Create and initialize the cross-attention block
                cross_attn_block = CrossAttentionBlock(self.model.config, layer.layer_idx).to(
                    self.model.device, dtype=self.model.dtype
                )
                cross_attn_block.self_attn.load_state_dict(copy.deepcopy(layer.self_attn.state_dict()))
                layer.cross_attn_block = cross_attn_block

                # Monkey-patch the forward method by binding our new method to the layer instance
                print(f"Injected CrossAttentionBlock into layer {layer.layer_idx}")
                layer.forward = modified_gemma_decoder_layer_forward.__get__(layer, Gemma3DecoderLayer)