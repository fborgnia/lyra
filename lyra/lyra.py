import torch
import torch.nn as nn

from .lyra_attention import LyraGemma3Attention

class GemmaInjector:
    def __init__(self, model):
        self.model = model

    def enable(self):
        """
        Replaces the self-attention module in global attention layers of the Gemma
        model with an identical, custom implementation (LyraGemma3Attention).
        This serves as a baseline for further modifications.
        """
        print("Starting baseline injection of LyraGemma3Attention...")
        for layer in self.model.model.layers:
            # Target only the global attention layers
            if not layer.self_attn.is_sliding:
                
                # 1. Create an instance of our custom attention class
                lyra_attn_module = LyraGemma3Attention(
                    config=layer.self_attn.config, 
                    layer_idx=layer.layer_idx
                ).to(self.model.device, dtype=self.model.dtype)
                
                # 2. Copy the weights and biases from the original module
                lyra_attn_module.load_state_dict(layer.self_attn.state_dict())
                
                # 3. Replace the original module with our new one
                layer.self_attn = lyra_attn_module
                
                print(f"Replaced attention module in layer {layer.layer_idx} with LyraGemma3Attention.")
        print("Baseline injection complete.")