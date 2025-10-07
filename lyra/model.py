import os
import torch
import torch.nn as nn
from transformers import Gemma3ForCausalLM, AutoTokenizer, AutoConfig
from .attention import LyraAttention
from typing import Optional, Tuple, Union, Dict, Any
from transformers.modeling_outputs import CausalLMOutputWithPast


class Lyra(Gemma3ForCausalLM):
    """
    Lyra is a Gemma 3 model that uses a custom attention mechanism.
    This class serves as a wrapper around the base Gemma model to allow for
    plug-and-play replacement of core components.
    """
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DEFAULT_MODEL_PATH = os.path.join(_project_root, 'models/gemma-3-1b-it')

    def __init__(self, pretrained_model_name_or_path=None, *model_args, **kwargs):
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = self.DEFAULT_MODEL_PATH

        # Load config and initialize the model structure from pretrained
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        super().__init__(config)

        # Load the pre-trained weights into this model
        temp_model = Gemma3ForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        self.load_state_dict(temp_model.state_dict())

        # Attach the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        # Replace the standard attention module with our custom one
        if hasattr(self, 'model') and hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                layer.self_attn = LyraAttention(config)

        self._freeze_base_model()

    def _freeze_base_model(self):
        """
        (Internal) Freezes all parameters of the base Gemma model.
        This is a placeholder for now. In later stages, we will unfreeze
        the parameters of our custom attention module.
        """
        for param in self.parameters():
            param.requires_grad = False

        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Lyra base model frozen. Trainable parameters: {num_trainable}")