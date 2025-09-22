import sys
import torch
import torch.nn as nn
from transformers import Gemma3ForCausalLM, AutoTokenizer
from torch_geometric.data import Data

# These will be implemented later, but we create placeholders for now.
from .gnn import EpisodicMemoryGNN
from .injection import MemoryInjectionLayer

class GemmaWithMemory(nn.Module):
    def __init__(self, model_path='./models/gemma-3-1b-it'):
        super().__init__()
        
        # 1. Load a frozen Gemma instruction-tuned model and tokenizer
        self.gemma = Gemma3ForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Freeze the base model parameters
        for param in self.gemma.parameters():
            param.requires_grad = False
        
        # 2. Store special token IDs for quick lookup
        self.start_of_turn_token_id = self.tokenizer.convert_tokens_to_ids('<start_of_turn>')
        self.end_of_turn_token_id = self.tokenizer.convert_tokens_to_ids('<end_of_turn>')

        # 3. Initialize GNN, Injection Layer, and an empty memory graph
        self.gnn = EpisodicMemoryGNN()
        self.injection_layer = MemoryInjectionLayer(self.gnn)
        hidden_size = self.gemma.config.hidden_size
        self.memory_graph = Data(x=torch.empty((0, hidden_size)))
        self.memory_graph.edge_index = torch.empty((2, 0), dtype=torch.long)
        print("Initialized empty memory graph.", file=sys.stderr)

        # 4. Configure the injection layer and state management
        self.capture_layer_idx = 8
        self.is_prompt_processing = False

        # Register the hook for memory capture. Injection is disabled for now.
        self.gemma.model.layers[self.capture_layer_idx].register_forward_hook(self._memory_capture_hook)

        
    def _memory_capture_hook(self, module, input, output):
        """
        This hook runs after the forward pass of the specified layer.
        It captures the output hidden state for later use in memory generation.
        """
        if self.is_prompt_processing:
            print(f"Running capture hook on layer {self.capture_layer_idx}...", file=sys.stdout)
            self._update_memory(output[0].detach())
            self.is_prompt_processing = False

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # The hooks will handle the memory operations automatically.
        return self.gemma(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def generate(self, input_ids, **kwargs):
        print(f"Generating Answer", file=sys.stdout)
        # 1. Set the flag to enable the capture hook for this prompt. the hook will reset it.
        self.is_prompt_processing = True
        
        # 2. Generate the response. The hook will be triggered automatically.
        outputs = self.gemma.generate(
            input_ids=input_ids,
            **kwargs
        )
        return outputs

    def _update_memory(self, prompt_hidden_state):
        """
        Internal method to update the memory graph using the captured hidden state.
        """
        
        print(f"Updating memory using hidden state from layer {self.capture_layer_idx}.", file=sys.stdout)
        
        # 1. Pool the prompt's hidden states to create a single summary vector.
        turn_summary_vector = torch.mean(prompt_hidden_state, dim=1)

        print(f"Created turn summary vector with shape: {turn_summary_vector.shape}", file=sys.stderr)

        # 2. Add the new summary vector as a node to the graph.
        existing_nodes = self.memory_graph.x
        num_existing_nodes = existing_nodes.shape[0]
        new_nodes = torch.cat([existing_nodes, turn_summary_vector], dim=0)
        
        # 3. Add an edge from the previous node to the new one.
        if num_existing_nodes > 0:
            prev_node_idx = num_existing_nodes - 1
            new_node_idx = num_existing_nodes
            new_edge = torch.tensor([[prev_node_idx], [new_node_idx]], dtype=torch.long)
            new_edge_index = torch.cat([self.memory_graph.edge_index, new_edge], dim=1)
        else:
            # This is the first node, so no new edges are needed.
            new_edge_index = self.memory_graph.edge_index
            
        self.memory_graph.x = new_nodes
        self.memory_graph.edge_index = new_edge_index
        print(f"Added new node. Graph now has {self.memory_graph.num_nodes} nodes.", file=sys.stderr)

        pass

