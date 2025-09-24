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
    an episodic memory graph. It overrides the `generate` method for inference and
    the `forward` method for training.
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

        # 4. Initialize GNN, Injection Layer, and an empty memory graph for INFERENCE
        self.gnn = EpisodicMemoryGNN()
        self.injection_layer = MemoryInjectionLayer(self.gnn)
        hidden_size = self.config.hidden_size
        self.memory_graph = Data(x=torch.empty((0, hidden_size)))
        self.memory_graph.edge_index = torch.empty((2, 0), dtype=torch.long)
        print("Initialized empty memory graph.", file=sys.stderr)

    def forward(self, input_ids=None, attention_mask=None, labels=None, inputs_embeds=None,
                context_turns_input_ids=None, current_turn_input_ids=None, **kwargs):
        """
        The forward pass handles both training and inference calls.
        - For training, it expects `context_turns_input_ids` and `current_turn_input_ids`.
        - For inference (called by `generate`), it expects standard arguments like `inputs_embeds`.
        """
        # --- Case 1: Training Path ---
        # Check for the presence of our custom training arguments.
        if context_turns_input_ids is not None and current_turn_input_ids is not None:
            # --- 1a. Simulate Conversation to Build Memory Graph ---
            training_memory_graph = Data(x=torch.empty((0, self.config.hidden_size), device=self.device))
            training_memory_graph.edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)

            for turn_input_ids in context_turns_input_ids:
                turn_embeddings = self.model.embed_tokens(turn_input_ids)
                turn_summary_vector = torch.mean(turn_embeddings, dim=1)
                
                existing_nodes = training_memory_graph.x
                num_existing_nodes = existing_nodes.shape[0]
                new_nodes = torch.cat([existing_nodes, turn_summary_vector], dim=0)
                
                if num_existing_nodes > 0:
                    prev_node_idx = num_existing_nodes - 1
                    new_node_idx = num_existing_nodes
                    new_edge = torch.tensor([[prev_node_idx], [new_node_idx]], dtype=torch.long, device=self.device)
                    new_edge_index = torch.cat([training_memory_graph.edge_index, new_edge], dim=1)
                else:
                    new_edge_index = training_memory_graph.edge_index
                
                training_memory_graph.x = new_nodes
                training_memory_graph.edge_index = new_edge_index

            # --- 1b. Perform Injection for the Current Turn ---
            initial_hidden_states = self.model.embed_tokens(current_turn_input_ids)
            query_vector = torch.mean(initial_hidden_states, dim=1)
            current_attention_mask = torch.ones_like(current_turn_input_ids)

            modified_hidden_states, modified_attention_mask = self.injection_layer(
                initial_hidden_states, current_attention_mask, training_memory_graph, query_vector
            )

            # --- 1c. Delegate for Loss Calculation ---
            return super().forward(
                input_ids=None,
                inputs_embeds=modified_hidden_states,
                attention_mask=modified_attention_mask,
                labels=labels,
                **kwargs
            )

        # --- Case 2: Inference Path ---
        # If training arguments are not present, assume it's an inference call from `generate`.
        # Simply pass all arguments to the parent class's forward method.
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def generate(self, input_ids, **kwargs):
        """
        Overrides the main generate method for INFERENCE.
        """
        # --- 1. Pre-computation for Memory Operations (Inference) ---
        if input_ids is not None:
            print("GemmaWithMemory: Pre-computation for memory operations.", file=sys.stdout)
            
            initial_hidden_states = self.model.embed_tokens(input_ids)
            query_vector = torch.mean(initial_hidden_states, dim=1)
            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)

            # --- Memory Injection (using stateful inference graph) ---
            modified_hidden_states, modified_attention_mask = self.injection_layer(
                initial_hidden_states, attention_mask, self.memory_graph, query_vector
            )
            num_injected_nodes = modified_hidden_states.shape[1] - initial_hidden_states.shape[1]
            if num_injected_nodes > 0:
                print(f"Injecting {num_injected_nodes} memory node(s) into the prompt.", file=sys.stdout)
            else:
                print("No memory nodes injected for this turn.", file=sys.stdout)
            
            # --- Memory Capture (updating stateful inference graph) ---
            self._update_memory(initial_hidden_states.detach())

            # --- 2. Delegate to the Original `generate` Method with Modified Inputs ---
            if 'attention_mask' in kwargs:
                del kwargs['attention_mask']
            return super().generate(
                input_ids=None,
                inputs_embeds=modified_hidden_states,
                attention_mask=modified_attention_mask,
                **kwargs
            )

        return super().generate(input_ids=input_ids, **kwargs)

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