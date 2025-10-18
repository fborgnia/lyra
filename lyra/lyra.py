import torch
import types
from typing import Optional

from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel, Gemma3DecoderLayer
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging

logger = logging.get_logger(__name__)

# 1. Define the new forward functions that will be injected.
#    For the baseline, it simply calls the original methods.

def lyra_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> BaseModelOutputWithPast:
    """
    This function will replace Gemma3TextModel.forward.
    For the baseline, it simply calls the original forward method.
    """
    # print("[Injector] In lyra_forward") # Uncomment for debugging
    
    # Call the original method that we saved on the model instance
    return self.original_text_model_forward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **kwargs,
    )

def lyra_decoder_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings_global: torch.Tensor,
    position_embeddings_local: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    output_attentions: Optional[bool] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    This function will replace Gemma3DecoderLayer.forward.
    For the baseline, it simply calls the original forward method for this layer.
    """
    # print(f"[Injector] In lyra_decoder_forward for layer {self.layer_idx}") # Uncomment for debugging

    # Call the original method for this specific layer
    return self.original_decoder_layer_forward(
        hidden_states=hidden_states,
        position_embeddings_global=position_embeddings_global,
        position_embeddings_local=position_embeddings_local,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )

# 2. Define the Injector class to perform the monkey-patching.

class GemmaInjector:
    def __init__(self, model):
        self.model = model

    def enable(self):
        """
        Replaces the forward methods of the Gemma model and its decoder layers
        with our new pass-through implementations.
        """
        print("Enabling Lyra baseline injector...")

        # --- Save and Patch Gemma3TextModel.forward ---
        # Save the original method directly onto the model instance
        self.model.model.original_text_model_forward = self.model.model.forward
        # Replace the instance's forward method with our new one
        self.model.model.forward = types.MethodType(lyra_forward, self.model.model)
        print("  - Patched Gemma3TextModel.forward")

        # --- Save and Patch Gemma3DecoderLayer.forward for each layer ---
        for i, layer in enumerate(self.model.model.layers):
            # Save the original method for this specific layer instance
            layer.original_decoder_layer_forward = layer.forward
            # Replace the instance's forward method
            layer.forward = types.MethodType(lyra_decoder_forward, layer)
        
        print(f"  - Patched Gemma3DecoderLayer.forward for {len(self.model.model.layers)} layers.")
        print("Lyra baseline injection complete.")