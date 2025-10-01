import os
import torch
import torch.nn as nn
from transformers import Gemma3ForCausalLM, AutoTokenizer, AutoConfig
from .episodic_memory import LyraDecoderLayer, MemoryInjectionBlock, MemoryArchivalBlock
import types
from typing import Optional, Tuple, Union, Dict, Any
from transformers.modeling_outputs import CausalLMOutputWithPast


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

        # Load config and initialize the model structure from pretrained
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        super().__init__(config)

        # Load the pre-trained weights into this model
        temp_model = Gemma3ForCausalLM.from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        self.load_state_dict(temp_model.state_dict())

        # Attach the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)

        # Add the memory archival block
        self.memory_archival_block = MemoryArchivalBlock()

        # Use a non-destructive, in-place modification of the decoder layers
        if hasattr(self, 'model') and hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                # 1. Add the memory injection block to the existing layer instance
                layer.memory_injection_block = MemoryInjectionBlock()
                
                # 2. Replace the layer's forward method with our custom one
                layer.forward = types.MethodType(LyraDecoderLayer.forward, layer)

        # Register a forward hook on the final layer norm to capture the last hidden state
        # This is now handled by the generate wrapper
        pass

    def generate(self, *args, **kwargs):
        """
        Wraps the original generate method to handle memory archival.
        A forward hook is temporarily registered to capture the final hidden state
        after the full generation is complete.
        """
        handle = None
        
        # This list will store the last hidden state from the final forward pass
        last_hidden_state_container = []

        def hook(module, input, output):
            # The hook will be called for each token generation step.
            # We only care about the last one, so we just keep overwriting.
            # The 'input' is a tuple, and the first element is the hidden state tensor.
            last_hidden_state_container.clear()
            last_hidden_state_container.append(input[0])

        if hasattr(self, 'model') and hasattr(self.model, 'norm'):
            handle = self.model.norm.register_forward_hook(hook)

        try:
            # Call the original generate method
            outputs = super().generate(*args, **kwargs)
        finally:
            # Always remove the hook afterwards
            if handle:
                handle.remove()

        # After generation, perform the memory archival with the captured hidden state
        if last_hidden_state_container:
            last_hidden_state = last_hidden_state_container[0]
            # We need to find the attention mask. A common way is to look for it in kwargs.
            attention_mask = kwargs.get('attention_mask', None)
            self.memory_archival_block(last_hidden_state, attention_mask)
        
        return outputs