import torch
import types
from typing import Optional

from transformers.models.gemma3.modeling_gemma3 import Gemma3TextModel, Gemma3DecoderLayer
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import logging

from .model import forward
from .decoder import forward as decoder_forward

logger = logging.get_logger(__name__)

class GemmaInjector:
    def __init__(self, model):
        self.model = model

    def enable(self):
        """
        Replaces the forward methods of the Gemma model and its decoder layers
        """
        print("Enabling Lyra baseline injector...")

        # --- Save and Patch Gemma3TextModel.forward ---
        # Save the original method directly onto the model instance
        self.model.model.original_text_model_forward = self.model.model.forward
        # Replace the instance's forward method with our new one
        self.model.model.forward = types.MethodType(forward, self.model.model)
        print("  - Patched Gemma3TextModel.forward")

        # --- Save and Patch Gemma3DecoderLayer.forward for each layer ---
        for i, layer in enumerate(self.model.model.layers):
            # Save the original method for this specific layer instance
            layer.original_decoder_layer_forward = layer.forward
            # Replace the instance's forward method
            layer.forward = types.MethodType(decoder_forward, layer)
        
        print(f"  - Patched Gemma3DecoderLayer.forward for {len(self.model.model.layers)} layers.")
        print("Lyra baseline injection complete.")