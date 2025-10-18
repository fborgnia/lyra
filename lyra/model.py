import torch
from typing import Optional

from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import logging

logger = logging.get_logger(__name__)

def forward(
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
    # Uncomment for debugging to confirm this path is being used
    # if 'lyra_past_key_values' in kwargs:
    #     print(f"[Injector] In forward, received lyra_past_key_values.")

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