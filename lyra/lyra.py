import torch
import types
from typing import Optional

from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel, Gemma3DecoderLayer
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging

from .causal_lm import forward as causal_lm_forward
from .model import forward
from .decoder import forward as decoder_forward

logger = logging.get_logger(__name__)

class GemmaInjector:
    def __init__(self, model):
        self.model = model
        total_layers = model.config.num_hidden_layers
        #lyra_layers = list(range(1, total_layers, 2)) # e.g., [1, 3, 5, 7, ...]
        #self.lyra_layer_indices = list(range(1, total_layers, 2))
        #self.lyra_layer_indices = [2,5,8,11,14,17,20,23]  # Manually specified Lyra layers
        #self.lyra_layer_indices = [5,11,17,23]
        self.lyra_layer_indices = [5,11,17,23]
        #self.lyra_layer_indices = lyra_layer_indices if lyra_layer_indices is not None else []

    def enable(self):
        """
        Replaces the forward methods of the Gemma model and its decoder layers
        """
        print("Enabling Lyra injector...")

        # --- Patch Gemma3ForCausalLM.forward ---
        # Replace the instance's forward method with our new one
        self.model.forward = types.MethodType(causal_lm_forward, self.model)
        print("  - Patched Gemma3ForCausalLM.forward")

        # --- Patch Gemma3TextModel.forward ---
        # Replace the instance's forward method with our new one
        self.model.model.forward = types.MethodType(forward, self.model.model)
        print("  - Patched Gemma3TextModel.forward")

        # --- Patch Gemma3DecoderLayer.forward for each layer ---
        for i, layer in enumerate(self.model.model.layers):
            if i in self.lyra_layer_indices:
                if layer.attention_type == "full_attention":
                    layer.attention_type = "full_cross_attention"
                elif layer.attention_type == "sliding_attention":
                    layer.attention_type = "sliding_cross_attention"
                
                print(f"  - Converted layer {i} to {layer.attention_type}.")
            # Replace the instance's forward method
            layer.forward = types.MethodType(decoder_forward, layer)
        
        print(f"  - Patched Gemma3DecoderLayer.forward for {len(self.model.model.layers)} layers.")
        print("Lyra injection complete.")