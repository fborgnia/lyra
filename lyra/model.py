import torch
import os
from transformers import Gemma3ForCausalLM, AutoTokenizer, AutoConfig
from .attention import LyraAttention
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache


class Lyra(Gemma3ForCausalLM):
    """
    Lyra is a stateful Gemma 3 model that uses a custom attention mechanism
    and manages a sliding KV cache across conversational turns.
    """
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DEFAULT_MODEL_PATH = os.path.join(_project_root, 'models/gemma-3-1b-it')

    def __init__(self, pretrained_model_name_or_path=None, *model_args, **kwargs):
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = self.DEFAULT_MODEL_PATH

        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        # The model must be configured to use the cache, or it will not be returned
        # in the forward pass.
        config.use_cache = True
        super().__init__(config)

        temp_model = Gemma3ForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        self.load_state_dict(temp_model.state_dict())

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        if hasattr(self, 'model') and hasattr(self.model, 'layers'):
            for i, layer in enumerate(self.model.layers):
                original_attn = layer.self_attn
                layer.self_attn = LyraAttention(self.config, layer_idx=i)
                layer.self_attn.load_state_dict(original_attn.state_dict(), strict=False)
        
        self.conversation_cache = None

    def generate(self, *args, **kwargs):
        """
        Stateful generate method that manages the KV cache across calls.
        """
        # If a conversation is in progress, inject the cache and create cache_position.
        if self.conversation_cache is not None:
            kwargs["past_key_values"] = self.conversation_cache
            
            # --- THE FIX: Create and pass the cache_position tensor ---
            # This tensor tells the model how long the cached sequence is.
            past_length = self.conversation_cache.get_seq_length()
            input_ids = kwargs.get("input_ids")
            if input_ids is None and len(args) > 0:
                input_ids = args[0]
            
            # The new position_ids should start from the end of the cached sequence
            kwargs["position_ids"] = torch.arange(
                past_length,
                past_length + input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device
            ).unsqueeze(0)

            kwargs["cache_position"] = torch.arange(
                past_length,
                past_length + input_ids.shape[1],
                dtype=torch.long,
                device=input_ids.device
            )
            # --- End of Fix ---
        # Force the generate method to return the dictionary-like ModelOutput
        kwargs["return_dict_in_generate"] = True
        # Ensure past_key_values are returned
        kwargs["output_scores"] = True 

        # Call the original generate method
        outputs = super().generate(*args, **kwargs)

        # Save the updated cache for the next turn.
        if hasattr(outputs, "past_key_values"):
            self.conversation_cache = outputs.past_key_values
        
        return outputs.sequences

