import torch
import torch.nn as nn
from transformers import Gemma3ForCausalLM, AutoTokenizer

# These will be implemented later, but we create placeholders for now.
# from .gnn import EpisodicMemoryGNN
# from .injection import MemoryInjectionLayer

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
        # self.gnn = EpisodicMemoryGNN(...)
        # self.injection_layer = MemoryInjectionLayer(...)
        self.memory_graph = None

        # TODO: Integrate the injection layer into the gemma model's layers
        # e.g., self.gemma.model.layers[10] = self.injection_layer

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # "Instruct-Aware" forward method
        # Check if <start_of_turn> is in the input and reset memory
        if self.start_of_turn_token_id in input_ids:
            print("New conversation detected. Resetting memory.")
            self.memory_graph = None
        
        # The actual forward pass will go through the modified Gemma model
        # which now includes the MemoryInjectionLayer
        return self.gemma(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def generate(self, input_ids, **kwargs):
        # Custom generate method to wrap the standard generate function
        
        # Request hidden states and a dictionary output for easier access
        kwargs['output_hidden_states'] = True
        kwargs['return_dict_in_generate'] = True

        # Store the length of the input prompt
        input_length = input_ids.shape[1]

        # Generate the response
        outputs = self.gemma.generate(input_ids=input_ids, **kwargs)
        
        # Check if the generation ended with the <end_of_turn> token.
        if outputs.sequences[0][-1] == self.end_of_turn_token_id:
             print("Turn complete. Triggering memory update.")
             # Pass the full output and the input length to the update method.
             self._update_memory(outputs, input_length)

        # Return only the generated sequences to maintain standard behavior
        return outputs.sequences

    def _update_memory(self, generated_outputs, input_length):
        """
        Internal method to update the memory graph with the latest turn.
        This implementation extracts the hidden states of the generated tokens.
        """
        # The `hidden_states` from `generate` is a tuple of tuples.
        # Outer tuple: one element for each generated token.
        # Inner tuple: one element for each layer's hidden state.
        # We want the hidden state from the last layer for each generated token.
        
        # `generated_outputs.hidden_states` contains states for generated tokens only.
        # Each element of the outer tuple corresponds to a single generation step.
        # Each of these elements is a tuple of all layer states for that step.
        # We take the last layer's state [-1] for each step.
        last_layer_hidden_states = [
            step_hidden_states[-1] for step_hidden_states in generated_outputs.hidden_states
        ]
        
        # Stack the states for all generated tokens into a single tensor.
        # Shape: [num_generated_tokens, batch_size, hidden_dim]
        turn_hidden_states = torch.stack(last_layer_hidden_states)
        
        # For now, we'll just print the shape to verify.
        # The batch dimension is in the middle, so we'll permute it to be more intuitive.
        # New shape: [batch_size, num_generated_tokens, hidden_dim]
        turn_hidden_states = turn_hidden_states.permute(1, 0, 2)
        
        print(f"Extracted hidden states for the turn with shape: {turn_hidden_states.shape}")

        # TODO:
        # 1. Pool the hidden states to create a single summary vector.
        # 2. Initialize the memory_graph if it's None.
        # 3. Add the new summary vector as a node to self.memory_graph.
        pass# Phase 2: GemmaWithMemory main class
