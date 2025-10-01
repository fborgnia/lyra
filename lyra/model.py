import os
import torch.nn as nn
from transformers import Gemma3ForCausalLM, AutoTokenizer, AutoConfig
from .episodic_memory import LyraDecoderLayer, MemoryInjectionBlock
import types

class Lyra(Gemma3ForCausalLM):
    """
    Lyra is a Gemma 3 model that integrates a memory injection block into its
    decoder layers.
    """
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DEFAULT_MODEL_PATH = os.path.join(_project_root, 'models/gemma-3-1b-it')

    def __init__(self, pretrained_model_name_or_path=None, *model_args, **kwargs):
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = self.DEFAULT_MODEL_PATH

        # Load config and initialize the model structure
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        super().__init__(config)

        # Load the pre-trained weights into this model
        # We create a temporary model to get the state_dict, then load it into self
        temp_model = Gemma3ForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        self.load_state_dict(temp_model.state_dict())

        # Attach the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        # Use a non-destructive, in-place modification of the decoder layers
        if hasattr(self, 'model') and hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                # 1. Add the memory injection block to the existing layer instance
                layer.memory_injection_block = MemoryInjectionBlock()
                
                # 2. Replace the layer's forward method with our custom one
                layer.forward = types.MethodType(LyraDecoderLayer.forward, layer)
