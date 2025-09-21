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
        print(f"Generating Answer")
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
        print(f"Updating memory")
        # The `hidden_states` from `generate` is a tuple of tuples.
        # Outer tuple: one element for each generated token.
        # Inner tuple: one element for each layer's hidden state.
        
        # At each generation step, the hidden state tensor has shape [batch, seq_len, hidden_dim].
        # The seq_len is > 1 for the first step and 1 for all subsequent steps.
        # We must select only the state for the last token in the sequence at each step.
        last_layer_hidden_states = [
            step_hidden_states[-1][:, -1:, :] for step_hidden_states in generated_outputs.hidden_states
        ]
        
        # Now, every tensor in the list has shape [batch, 1, hidden_dim].
        # We can concatenate them along the sequence dimension (dim=1).
        turn_hidden_states = torch.cat(last_layer_hidden_states, dim=1)
        
        # The resulting shape is already [batch_size, num_generated_tokens, hidden_dim].
        
        print(f"Extracted hidden states for the turn with shape: {turn_hidden_states.shape}")

        # 1. Pool the hidden states to create a single summary vector.
        # We use mean pooling across the sequence of generated tokens.
        turn_summary_vector = torch.mean(turn_hidden_states, dim=1)

        print(f"Created turn summary vector with shape: {turn_summary_vector.shape}")

        # TODO:
        # 2. Initialize the memory_graph if it's None.
        # 3. Add the new summary vector as a node to self.memory_graph.
        pass
        
