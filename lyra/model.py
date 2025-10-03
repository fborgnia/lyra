import os
import torch
import torch.nn as nn
from transformers import Gemma3ForCausalLM, AutoTokenizer, AutoConfig
from .memory import LyraDecoderLayer, MemoryInjectionBlock, MemoryArchivalBlock, EpisodicMemoryStore
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
    DEFAULT_MEMORY_WEIGHTS_PATH = os.path.join(_project_root, 'models', 'lyra_memory_heads.pth')

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

        # Create and share the central memory store
        self.memory_store = EpisodicMemoryStore()

        # Add the memory archival block
        self.memory_archival_block = MemoryArchivalBlock(self.memory_store)

        # Use a non-destructive, in-place modification of the decoder layers
        if hasattr(self, 'model') and hasattr(self.model, 'layers'):
            for layer in self.model.layers:
                # 1. Add the memory injection block to the existing layer instance
                layer.memory_injection_block = MemoryInjectionBlock(config, self.memory_store)
                
                # 2. Add the post-memory layer norm
                layer.post_memory_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

                # 3. Replace the layer's forward method with our custom one
                layer.forward = types.MethodType(LyraDecoderLayer.forward, layer)

        if os.path.exists(self.DEFAULT_MEMORY_WEIGHTS_PATH):
            print(f"Warning: Weights path found: {self.DEFAULT_MEMORY_WEIGHTS_PATH}")
            # Load the state dictionary, ensuring it's on the correct device.
            memory_weights = torch.load(self.DEFAULT_MEMORY_WEIGHTS_PATH, map_location=self.device)
            # Load the weights into the model. `strict=False` is essential because
            # we are only loading a subset of the model's parameters (the trained ones).
            self.load_state_dict(memory_weights, strict=False)
            print(f"Successfully loaded memory weights from {self.DEFAULT_MEMORY_WEIGHTS_PATH}")

        self._freeze_base_model()
    
    def _freeze_base_model(self):
        """
        (Internal) Freezes all parameters of the base Gemma model, leaving only the
        memory-related components (injection blocks and layer norms) trainable.
        """
        # First, freeze all parameters
        for param in self.parameters():
            param.requires_grad = False

        # Then, unfreeze only the parameters of our custom memory modules
        for layer in self.model.layers:
            if hasattr(layer, 'memory_injection_block'):
                for param in layer.memory_injection_block.parameters():
                    param.requires_grad = True
            if hasattr(layer, 'post_memory_layernorm'):
                for param in layer.post_memory_layernorm.parameters():
                    param.requires_grad = True
        
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Lyra base model frozen. Trainable parameters: {num_trainable}")
    
    def generate(self, *args, **kwargs):
        """
        Wraps the original generate method to handle memory archival.
        First, it generates the token sequence. Then, it performs a single
        forward pass with the complete sequence to reliably capture the final
        hidden states for archival.
        """
        # Step 1: Generate token sequences.
        # Ensure no conflicting flags are passed from previous attempts.
        kwargs.pop('output_hidden_states', None)
        kwargs.pop('return_dict_in_generate', None)
        
        generated_outputs = super().generate(*args, **kwargs)

        # The output of generate can be a tensor or a dict-like object.
        output_sequences = generated_outputs if isinstance(generated_outputs, torch.Tensor) else generated_outputs.sequences

        # Step 2: Perform a single forward pass with the full generated sequence to get the final state.
        # We create an attention mask where all tokens are attended to.
        attention_mask = torch.ones_like(output_sequences)

        # We don't need to track gradients for this archival pass.
        with torch.no_grad():
            model_outputs = self.forward(
                input_ids=output_sequences,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # The `hidden_states` is a tuple of tensors (one for each layer + embeddings).
        # We want the output of the last decoder layer.
        last_hidden_state = model_outputs.hidden_states[-1]

        # Step 3: Archive the memory. The tensors are now guaranteed to be aligned.
        self.memory_archival_block(last_hidden_state, attention_mask)

        # Step 4: Return the original generated sequences, maintaining compatibility.
        return generated_outputs