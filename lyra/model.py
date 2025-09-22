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

        # 3. Initialize GNN, Injection Layer, and internal memory state
        self.gnn = EpisodicMemoryGNN()
        self.injection_layer = MemoryInjectionLayer(self.gnn)
        self.memory_graph = None

        # 4. Configure the injection layer and state management
        self.injection_layer_idx = 8
        self.is_prompt_processing = False
        self.last_layer_8_output = None

        # TODO: Register the hooks for memory injection and generation.
        # For now, we will call it manually in the forward pass for demonstration.
        self.gemma.model.layers[self.injection_layer_idx].register_forward_pre_hook(self._injection_pre_hook)
        self.gemma.model.layers[self.injection_layer_idx].register_forward_hook(self._memory_capture_hook)

    def _memory_capture_hook(self, module, input, output):
        """
        This hook runs after the forward pass of the specified layer (layer 8).
        It captures the output hidden state for later use in memory generation.
        """
        # The output of a decoder layer is a tuple; the hidden state is the first element.
        # We only want to capture this during the initial prompt processing phase.
        if self.is_prompt_processing:
            print(f"Running capture hook on layer {self.injection_layer_idx}...", file=sys.stderr)
            self.last_layer_8_output = output[0].detach()

    def _injection_pre_hook(self, module, args):
        """
        This hook runs before the forward pass of the specified layer (layer 8).
        It injects memory into the hidden states if conditions are met.
        """
        if self.is_prompt_processing and self.memory_graph is not None and self.memory_graph.num_nodes > 0:
            print(f"Running injection pre-hook on layer {self.injection_layer_idx}...", file=sys.stderr)
            
            # The input arguments to a decoder layer are a tuple.
            # We need to unpack them, modify them, and return a new tuple.
            hidden_states, attention_mask = args[0], args[1]
            
            # 1. Create a query vector from the incoming hidden states
            query_vector = torch.mean(hidden_states, dim=1)
            
            # 2. Use the existing injection_layer to get modified states and mask
            modified_hidden_states, modified_attention_mask = self.injection_layer(
                hidden_states, attention_mask, self.memory_graph, query_vector
            )
            
            # 3. Deactivate the flag after both capture and injection are done for the prompt.
            self.is_prompt_processing = False
            
            # 4. Re-package the arguments for the layer's forward method.
            # The other arguments (position_ids, etc.) are preserved.
            new_args = (modified_hidden_states, modified_attention_mask) + args[2:]
            return new_args
        
        # If conditions are not met, do nothing.
        return args

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # "Instruct-Aware" forward method
        # With the hook-based approach, this forward method is no longer needed for injection.
        # We will simplify it to a standard forward pass.
        # The hooks will handle the memory operations automatically.
        return self.gemma(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def generate(self, input_ids, **kwargs):
        print(f"Generating Answer", file=sys.stderr)
        # Custom generate method to wrap the standard generate function

        # 1. Set the flag to enable the hooks for this prompt.
        self.is_prompt_processing = True
        
        # 2. Generate the response. The hooks will be triggered automatically.
        outputs = self.gemma.generate(
            input_ids=input_ids,
            **kwargs
        )
        
        # 3. Check if the generation ended with the <end_of_turn> token.
        # The output is a tensor of token IDs, shape [batch_size, sequence_length].
        if outputs[0][-1] == self.end_of_turn_token_id:
             print("Turn complete. Triggering memory update.", file=sys.stderr)
             # The memory capture hook will have saved the required hidden state.
             self._update_memory()

        # Return only the generated sequences to maintain standard behavior
        return outputs

    def _update_memory(self):
        """
        Internal method to update the memory graph using the captured hidden state from layer 8.
        """
        # This will be updated to use self.last_layer_8_output from the capture hook.
        if self.last_layer_8_output is None:
            print("Warning: _update_memory called but no hidden state was captured.", file=sys.stderr)
            return

        prompt_hidden_state = self.last_layer_8_output
        print(f"Updating memory using hidden state from layer {self.injection_layer_idx}.", file=sys.stderr)
        
        # 1. Pool the prompt's hidden states to create a single summary vector.
        turn_summary_vector = torch.mean(prompt_hidden_state, dim=1)

        print(f"Created turn summary vector with shape: {turn_summary_vector.shape}", file=sys.stderr)

        # 2. Initialize the memory_graph if it's None.
        if self.memory_graph is None:
            # This is the first node in the graph.
            self.memory_graph = Data(x=turn_summary_vector)
            # Edges will be added from the second node onwards.
            self.memory_graph.edge_index = torch.empty((2, 0), dtype=torch.long)
            print("Initialized memory graph with the first node.", file=sys.stderr)
        else:
            # 3. Add the new summary vector as a node to self.memory_graph.
            # Add the new node's features.
            existing_nodes = self.memory_graph.x
            new_nodes = torch.cat([existing_nodes, turn_summary_vector], dim=0)
            
            # Add an edge from the previous node to the new node.
            prev_node_idx = existing_nodes.shape[0] - 1
            new_node_idx = prev_node_idx + 1
            new_edge = torch.tensor([[prev_node_idx], [new_node_idx]], dtype=torch.long)
            new_edge_index = torch.cat([self.memory_graph.edge_index, new_edge], dim=1)
            
            # Update the graph object.
            self.memory_graph.x = new_nodes
            self.memory_graph.edge_index = new_edge_index
            print(f"Added new node. Graph now has {self.memory_graph.num_nodes} nodes.", file=sys.stderr)

        # Reset the captured state to prevent re-use.
        self.last_layer_8_output = None
        pass

