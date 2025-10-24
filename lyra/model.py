import torch
from typing import Optional

from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.utils import ModelOutput, TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.generic import check_model_inputs
from transformers.processing_utils import Unpack

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
    # --- Lyra Arguments ---
    position_embeddings_lyra: Optional[torch.Tensor] = None,
    lyra_attention_mask: Optional[torch.Tensor] = None,
    lyra_past_key_values: Optional[Cache] = None,
    lyra_position_ids: Optional[torch.LongTensor] = None,
    lyra_cache_position: Optional[torch.LongTensor] = None,
    # --- End Lyra Arguments ---
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None and not self.training:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )
    
    # Calculate cache_position for the Lyra stream
    if lyra_past_key_values is not None:
        lyra_past_seen_tokens = lyra_past_key_values.get_seq_length()
        lyra_cache_position = torch.arange(
            lyra_past_seen_tokens,
            lyra_past_seen_tokens + inputs_embeds.shape[1],
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    if lyra_cache_position is not None:
        lyra_position_ids = lyra_cache_position.unsqueeze(0)

    # Create a cache_position_mapping for different attention types
    cache_position_mapping = {
        "full_attention": cache_position,
        "sliding_attention": cache_position,
        "full_cross_attention": lyra_cache_position,
        "sliding_cross_attention": lyra_cache_position,
    }

    # Create a position_id map for different attention types
    position_ids_mapping = {
        "full_attention": position_ids,
        "sliding_attention": position_ids,
        "full_cross_attention": lyra_position_ids,
        "sliding_cross_attention": lyra_position_ids,
    }

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        lyra_mask_kwargs = {
            "config": self.config,
            "input_embeds": inputs_embeds,
            #"attention_mask": attention_mask, 
            "attention_mask": None, 
            "cache_position": lyra_cache_position,
            "past_key_values": lyra_past_key_values,
            "position_ids": lyra_position_ids, 
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
            "sliding_attention": create_sliding_window_causal_mask(**mask_kwargs),
            "full_cross_attention": create_causal_mask(**lyra_mask_kwargs),  # to be filled if lyra_past_key_values exists
            "sliding_cross_attention": create_sliding_window_causal_mask(**lyra_mask_kwargs),
        }

    # embed positions
    hidden_states = inputs_embeds

    # create position embeddings mapping to be shared across the decoder layers
    position_embeddings_mapping = {
        "full_attention": self.rotary_emb(hidden_states, position_ids_mapping["full_attention"]),
        "sliding_attention": self.rotary_emb_local(hidden_states, position_ids_mapping["sliding_attention"]),
        "full_cross_attention": self.rotary_emb(hidden_states, position_ids_mapping["full_cross_attention"]),
        "sliding_cross_attention": self.rotary_emb_local(hidden_states, position_ids_mapping["sliding_cross_attention"]),
    }

    # create a past_key_values mapping, this i need to refactor to simplify the entire logic of this forward pass
    past_key_values_mapping = {
        "full_attention": past_key_values,
        "sliding_attention": past_key_values,
        "full_cross_attention": lyra_past_key_values,
        "sliding_cross_attention": lyra_past_key_values,
    }

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings_mapping[decoder_layer.attention_type],
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_ids=position_ids_mapping[decoder_layer.attention_type],
            past_key_values=past_key_values_mapping[decoder_layer.attention_type],
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position_mapping[decoder_layer.attention_type],
            **kwargs,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )