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

        # Load config and initialize the model structure
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        super().__init__(config)

        # Load the pre-trained weights into this model
        # We create a temporary model to get the state_dict, then load it into self
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

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # Ensure output_hidden_states is True to get the hidden states
        # We also need to preserve the original value of output_hidden_states
        original_output_hidden_states = output_hidden_states
        output_hidden_states = True

        # Call the original forward method
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        # After the forward pass, call the memory archival block
        # The last hidden state is the one before the final layer norm and LM head
        last_hidden_state = outputs.hidden_states[-1] if return_dict else outputs[2][-1]
        self.memory_archival_block(last_hidden_state, attention_mask)

        # If the user did not originally want hidden states, we remove them from the output
        if not original_output_hidden_states and return_dict:
            outputs.hidden_states = None
        
        return outputs