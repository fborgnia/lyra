import sys
import torch
import torch.nn as nn
from transformers import Gemma3ForCausalLM, AutoTokenizer
from torch_geometric.data import Data

from .gnn import EpisodicMemoryGNN
from .injection import MemoryInjectionLayer

class GemmaWithMemory(Gemma3ForCausalLM):
    """
    A self-contained Gemma model that inherits from Gemma3ForCausalLM and integrates
    an episodic memory graph. It overrides the `generate` method to perform
    memory operations in a stable and controlled manner.
    """
    def __init__(self, model_path='./models/gemma-3-1b-it'):
        # 1. Load the pretrained Gemma3ForCausalLM model using the recommended 'eager' attention.
        base_model = Gemma3ForCausalLM.from_pretrained(model_path, attn_implementation="eager")
        
        # Initialize the parent class with the loaded model's config
        super().__init__(base_model.config)
        # Load the state dict from the loaded model
        self.load_state_dict(base_model.state_dict())
        
        # 2. Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.end_of_turn_token_id = self.tokenizer.encode('<end_of_turn>', add_special_tokens=False)[0]

        # 3. Freeze the base model parameters
        for param in self.parameters():
            param.requires_grad = False

        # 4. Initialize GNN, Injection Layer, and an empty memory graph
        self.gnn = EpisodicMemoryGNN()
        self.injection_layer = MemoryInjectionLayer(self.gnn)
        hidden_size = self.config.hidden_size
        self.memory_graph = Data(x=torch.empty((0, hidden_size)))
        self.memory_graph.edge_index = torch.empty((2, 0), dtype=torch.long)
        print("Initialized empty memory graph.", file=sys.stderr)

    def generate(self, input_ids, **kwargs):
        """
        Overrides the main generate method to perform a pre-computation step
        for memory injection and archival before delegating to the original generation logic.
        """
        # --- 1. Pre-computation for Memory Operations ---
        # This block runs exactly once per top-level generate call.
        if input_ids is not None:
            print("GemmaWithMemory: Pre-computation for memory operations.", file=sys.stdout)
            
            # a. Manually compute the initial hidden states from the input_ids.
            initial_hidden_states = self.model.embed_tokens(input_ids)
            
            # b. Prepare a query vector from the user's prompt.
            query_vector = torch.mean(initial_hidden_states, dim=1)

            # c. Prepare a corresponding attention mask.
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            # --- Memory Injection ---
            # This happens first, as requested. The injection layer will use the query
            # to retrieve relevant memories and prepend them to the hidden states.
            # On the first turn, the graph is empty, and this will return the inputs unmodified.
            modified_hidden_states, modified_attention_mask = self.injection_layer(
                initial_hidden_states, attention_mask, self.memory_graph, query_vector
            )

            # --- Memory Capture ---
            # We capture the memory of the *original* prompt, before injection,
            # to avoid archiving the injected memory itself.
            self._update_memory(initial_hidden_states.detach())

            # --- 2. Delegate to the Original `generate` Method with Modified Inputs ---
            # We pass the `modified_hidden_states` (as `inputs_embeds`) and the
            # `modified_attention_mask` to the original generate method.
            if 'attention_mask' in kwargs:
                del kwargs['attention_mask']
            return super().generate(
                input_ids=None, # Must be None when using inputs_embeds
                inputs_embeds=modified_hidden_states,
                attention_mask=modified_attention_mask,
                **kwargs
            )

        # If input_ids are None (which happens in recursive calls inside generate),
        # just delegate directly to the parent method.
        return super().generate(input_ids=input_ids, **kwargs)

    # This method is no longer needed as we are not modifying the generation loop.
    # def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
    #     ...

    def _update_memory(self, prompt_hidden_state):
        print(f"Updating memory...", file=sys.stdout)
        turn_summary_vector = torch.mean(prompt_hidden_state, dim=1)
        existing_nodes = self.memory_graph.x
        num_existing_nodes = existing_nodes.shape[0]
        new_nodes = torch.cat([existing_nodes, turn_summary_vector], dim=0)
        
        if num_existing_nodes > 0:
            prev_node_idx = num_existing_nodes - 1
            new_node_idx = num_existing_nodes
            new_edge = torch.tensor([[prev_node_idx], [new_node_idx]], dtype=torch.long)
            new_edge_index = torch.cat([self.memory_graph.edge_index, new_edge], dim=1)
        else:
            new_edge_index = self.memory_graph.edge_index
            
        self.memory_graph.x = new_nodes
        self.memory_graph.edge_index = new_edge_index
        print(f"Added new node. Graph now has {self.memory_graph.num_nodes} nodes.", file=sys.stderr)