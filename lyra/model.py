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
        self.memory_layer_idx = 8
        self.is_prompt_processing_capture = False
        self.is_prompt_processing_injection = False

        # Register the hook for memory capture. Injection is disabled for now.
        self.gemma.model.layers[self.memory_layer_idx].register_forward_pre_hook(self._memory_injection_hook)
        self.gemma.model.layers[self.memory_layer_idx].register_forward_hook(self._memory_capture_hook)

    def _memory_injection_hook(self, module, args):
        """
        This hook runs BEFORE the forward pass of the specified layer.
        It injects memory into the hidden states if conditions are met.
        """
        if self.is_prompt_processing_injection and len(args) >= 3 :
            print(f"Running injection pre-hook on layer {self.memory_layer_idx}...", file=sys.stdout)
            # The input arguments to a decoder layer are a tuple.
            # We need to unpack them, modify them, and return a new tuple.
            hidden_states, attention_mask, position_ids = args[0], args[1], args[2]
            
            # 1. Create a query vector from the incoming hidden states
            query_vector = torch.mean(hidden_states, dim=1)
            
            # 2. Use the existing injection_layer to get modified states and mask
            modified_hidden_states, modified_attention_mask = self.injection_layer(
                hidden_states, attention_mask, self.memory_graph, query_vector
            )
            
            # 3. Create a new position_ids tensor for the modified sequence.
            memory_position = torch.zeros((1, 1), dtype=position_ids.dtype, device=position_ids.device)
            modified_position_ids = torch.cat([memory_position, position_ids], dim=1)

            # 4. Re-package the arguments for the layer's forward method.
            new_args = (modified_hidden_states, modified_attention_mask, modified_position_ids) + args[3:]
            self.is_prompt_processing_injection = False
            return new_args

        # If conditions are not met, do nothing.
        return args

    def _memory_capture_hook(self, module, input, output):
        """
        This hook runs before the forward pass of the specified layer (layer 8).
        It injects memory into the hidden states if conditions are met.
        """
        if self.is_prompt_processing_capture :
            print(f"Running memory hook on layer {self.memory_layer_idx}...", file=sys.stdout)
            self._update_memory(output[0].detach())
            self.is_prompt_processing_capture = False
            

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # The hooks will handle the memory operations automatically.
        return self.gemma(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def generate(self, input_ids, **kwargs):
        print(f"Generating Answer", file=sys.stdout)
        # 1. Set the flag to enable the capture hook for this prompt. the hook will reset it.
        self.is_prompt_processing_injection = True
        self.is_prompt_processing_capture = True
    
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
        
        print(f"Updating memory using hidden state from layer {self.memory_layer_idx}.", file=sys.stdout)
        
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

