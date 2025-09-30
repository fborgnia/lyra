import sys
import torch
import torch.nn as nn
from transformers import Gemma3ForCausalLM, AutoTokenizer
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
from transformers.masking_utils import create_causal_mask
from pathlib import Path

from .decoder import LyraDecoderLayer

class Lyra(Gemma3ForCausalLM):
    """
    A self-contained Gemma model that inherits from Gemma3ForCausalLM and integrates
    an episodic memory buffer. It overrides the `forward` method to inject memory
    at each layer.
    """
    def __init__(self, model_path='./models/gemma-3-1b-it', num_memory_layers: int = 4):
        # 1. Load the pretrained Gemma3ForCausalLM model and tokenizer
        base_model = Gemma3ForCausalLM.from_pretrained(model_path, attn_implementation="eager")
        super().__init__(base_model.config)
        self.load_state_dict(base_model.state_dict())
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 2. Replace standard decoder layers with LyraDecoderLayers
        self.model.layers = nn.ModuleList(
            [LyraDecoderLayer(self.config, i) if i < num_memory_layers else Gemma3DecoderLayer(self.config, i)
             for i in range(len(self.model.layers))]
        )
        self.num_memory_layers = num_memory_layers

        # 3. Initialize an enhanced memory structure
        self.memory_buffer = []
        print(f"Initialized Lyra with {num_memory_layers} memory layers and an empty memory buffer.", file=sys.stderr)


    def forward(self, input_ids, attention_mask=None, position_ids=None, past_key_values=None, inputs_embeds=None, labels=None, use_cache=None, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        """
        The forward pass handles both training and inference calls.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        # The `generate` function does not pass `cache_position`, so we have to create it
        if "cache_position" not in kwargs:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            kwargs["cache_position"] = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        
        # 4d mask is passed through the layers
        # Copied from `Gemma3TextModel.forward`
        causal_attention_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=kwargs["cache_position"],
            past_key_values=past_key_values,
        )

        hidden_states = inputs_embeds * (self.config.hidden_size**0.5)
        
        # Pre-compute rotary embeddings
        position_embeddings_global = self.model.rotary_emb(hidden_states, position_ids=position_ids)
        position_embeddings_local = self.model.rotary_emb_local(hidden_states, position_ids=position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        
        previous_layer_query = None

        for idx, decoder_layer in enumerate(self.model.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if isinstance(decoder_layer, LyraDecoderLayer):
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    retriever_attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    memory_buffer=self.memory_buffer,
                    previous_layer_query=previous_layer_query,
                    **kwargs,
                )
                hidden_states = layer_outputs[0]
                previous_layer_query = layer_outputs[1]

            else: # Standard Gemma3DecoderLayer
                 layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    **kwargs,
                )
                 hidden_states = layer_outputs[0]

            if output_attentions:
                if isinstance(decoder_layer, LyraDecoderLayer):
                    all_self_attns += (layer_outputs[2],)
                else:
                    all_self_attns += (layer_outputs[1],)

        hidden_states = self.model.norm(hidden_states)
        
        if use_cache:
            next_decoder_cache = past_key_values

        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        from transformers.modeling_outputs import CausalLMOutputWithPast
        
        if not return_dict:
            return (logits,) + (next_decoder_cache,) + all_hidden_states + all_self_attns

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


    def _update_memory(self, input_ids, attention_mask):
        """
        Stores the final hidden state and attention mask for a given turn.
        """
        print(f"Updating memory...", file=sys.stdout)
        with torch.no_grad():
            # Create position_ids on the fly
            batch_size, seq_length = input_ids.shape
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)

            # Perform a forward pass to get the final hidden state
            # We call the base model's forward pass here, not the overridden one
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True,
            )
            # Get the last hidden state from the base model part
            final_hidden_state = outputs.hidden_states[-1]

        # Store the hidden state and mask in the memory list
        self.memory_buffer.append({
            "hidden_state": final_hidden_state.cpu(),
            "attention_mask": attention_mask.cpu(),
        })
        print(f"Added new node. Buffer now has {len(self.memory_buffer)} nodes.", file=sys.stderr)
