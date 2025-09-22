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

        # TODO: Integrate the injection layer into the gemma model's layers
        # For now, we will call it manually in the forward pass for demonstration.

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # "Instruct-Aware" forward method
        print("Running forward pass...", file=sys.stdout)
        
        # Get the initial token embeddings for the input prompt.
        # These are the embeddings that the model expects as `inputs_embeds`.
        prompt_embeds = self.gemma.get_input_embeddings()(input_ids)

        if self.memory_graph is not None and self.memory_graph.num_nodes > 0:
            print("Querying memory graph...", file=sys.stdout)
            # 1. Create a query vector. For this, we need a richer representation,
            # so we run a separate forward pass to get the last hidden state.
            with torch.no_grad():
                prompt_outputs = self.gemma(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                prompt_hidden_states = prompt_outputs.hidden_states[-1]
            query_vector = torch.mean(prompt_hidden_states, dim=1)
            
            # 2. Pass the initial embeddings (not last hidden state) to the injection layer.
            modified_embeds, attention_mask = self.injection_layer(
                prompt_embeds, attention_mask, self.memory_graph, query_vector
            )
        
        # Return the appropriate embeddings and attention mask.
        if 'modified_embeds' in locals():
            return modified_embeds, attention_mask
        else:
            return prompt_embeds, attention_mask

    def generate(self, input_ids, **kwargs):
        print(f"Generating Answer", file=sys.stderr)
        # Custom generate method to wrap the standard generate function

        # 1. Call the forward pass to get potentially memory-infused embeddings.
        # The forward pass handles the memory retrieval and injection logic.
        final_prompt_embeds, final_attention_mask = self.forward(input_ids, **kwargs)
        
        # BUGFIX: The 'attention_mask' in kwargs is for the original input_ids.
        # We must use the 'final_attention_mask' returned by the forward pass,
        # as it may have been modified (e.g., by prepending memory).
        # We also remove it from kwargs to avoid passing it twice to `gemma.generate`.
        if 'attention_mask' in kwargs:
            del kwargs['attention_mask']

        # 2. Generate the response using the (potentially modified) embeddings.
        print(f"Shape of inputs_embeds: {final_prompt_embeds.shape}", file=sys.stderr)
        print(f"Shape of attention_mask: {final_attention_mask.shape}", file=sys.stderr)
        outputs = self.gemma.generate(
            inputs_embeds=final_prompt_embeds, 
            attention_mask=final_attention_mask, 
            **kwargs
        )
        
        # 3. Check if the generation ended with the <end_of_turn> token.
        # The output is a tensor of token IDs, shape [batch_size, sequence_length].
        if outputs[0][-1] == self.end_of_turn_token_id:
             print("Turn complete. Triggering memory update.", file=sys.stderr)
             # To create the memory, we need the prompt's hidden states.
             # We run a forward pass on the original input_ids to get them.
             with torch.no_grad():
                prompt_outputs = self.gemma(input_ids=input_ids, output_hidden_states=True)
                prompt_hidden_states = prompt_outputs.hidden_states[-1]
             self._update_memory(prompt_hidden_states)

        # Return only the generated sequences to maintain standard behavior
        return outputs

    def _update_memory(self, prompt_hidden_states):
        """
        Internal method to update the memory graph using the prompt's hidden states.
        """
        print(f"Updating memory using the prompt's context.", file=sys.stderr)
        
        # 1. Pool the prompt's hidden states to create a single summary vector.
        # We use mean pooling across the sequence of prompt tokens.
        turn_summary_vector = torch.mean(prompt_hidden_states, dim=1)

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

        # TODO:
        # The next step will be to use this graph.
        pass

