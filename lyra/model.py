import sys
import torch
import torch.nn as nn
from transformers import Gemma3ForCausalLM, AutoTokenizer
from pathlib import Path

from .gnn import EpisodicMemoryGNN
from .injection import MemoryInjectionLayer

class GemmaWithMemory(Gemma3ForCausalLM):
    """
    A self-contained Gemma model that inherits from Gemma3ForCausalLM and integrates
    an episodic memory graph. It overrides the `generate` method for inference and
    the `forward` method for training the GNN with a triplet loss.
    """
    def __init__(self, model_path='./models/gemma-3-1b-it', gnn_weights_path='./models/gnn_semantic_alignment.pth'):
        # 1. Load the pretrained Gemma3ForCausalLM model and tokenizer
        base_model = Gemma3ForCausalLM.from_pretrained(model_path, attn_implementation="eager")
        super().__init__(base_model.config)
        self.load_state_dict(base_model.state_dict())
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 2. Freeze the base model parameters
        for param in self.parameters():
            param.requires_grad = False

        # 3. Initialize GNN and load trained weights
        self.gnn = EpisodicMemoryGNN(self.config, self.model.embed_tokens)
        gnn_weights_file = Path(gnn_weights_path)
        if gnn_weights_file.is_file():
            print(f"Loading trained GNN weights from {gnn_weights_path}", file=sys.stderr)
            self.gnn.load_state_dict(torch.load(gnn_weights_path))
        else:
            print(f"Warning: No trained GNN weights found at {gnn_weights_path}. GNN is using initial weights.", file=sys.stderr)

        self.injection_layer = MemoryInjectionLayer()

        # 5. Initialize an enhanced memory structure
        self.memory_graph = []
        print("Initialized empty memory graph.", file=sys.stderr)

        # 6. Define the loss function for GNN training
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2)

    def forward(self, context_input_ids=None, query_input_ids=None, **kwargs):
        """
        The forward pass handles both training and inference calls.
        """
        # --- Case 1: Training Path ---
        if context_input_ids is not None and query_input_ids is not None:
            with torch.no_grad():
                context_embeddings = self.model.embed_tokens(context_input_ids)
                query_embeddings = self.model.embed_tokens(query_input_ids)
                context_vectors = torch.mean(context_embeddings, dim=1)
                query_vectors = torch.mean(query_embeddings, dim=1)

            anchor = self.gnn.query_projection(query_vectors)
            positive = self.gnn.query_projection(context_vectors)

            batch_size = anchor.shape[0]
            if batch_size < 2:
                return {"loss": torch.tensor(0.0, device=self.device, requires_grad=True)} # Cannot compute loss for batch size < 2
            
            negative_indices = torch.arange(batch_size, device=self.device).roll(shifts=1, dims=0)
            negative = positive[negative_indices]

            loss = self.triplet_loss(anchor, positive, negative)
            return {"loss": loss}

        # --- Case 2: Inference Path ---
        return super().forward(**kwargs)

    def generate(self, input_ids, **kwargs):
        """
        Overrides the main generate method for INFERENCE.
        Injects text-based memory before calling the base model's generate method.
        """
        if input_ids is None:
            return super().generate(**kwargs)

        print("GemmaWithMemory: Pre-computation for memory operations.", file=sys.stdout)
        
        # --- 1. Inject retrieved memory to create a new set of input_ids ---
        # Pass the necessary components to the stateless injection layer
        modified_input_ids, modified_attention_mask = self.injection_layer(
            self.gnn,
            self.memory_graph,
            self.tokenizer,
            input_ids,
            kwargs.get("attention_mask", torch.ones_like(input_ids))
        )
        
        # Remove 'attention_mask' from kwargs to avoid passing it twice
        kwargs.pop("attention_mask", None)
        # --- 2. Delegate to the Original `generate` Method ---
        # We use the new, memory-infused input_ids to generate a response
        generated_outputs = super().generate(
            input_ids=modified_input_ids,
            attention_mask=modified_attention_mask,
            **kwargs
        )

        # --- 3. Update memory AFTER generation is complete ---
        # We store the original prompt (before injection) as a memory.
        self._update_memory(input_ids)

        return generated_outputs

    def _update_memory(self, original_input_ids):
        """
        Stores the summary vector and original input_ids for a given turn.
        """
        print(f"Updating memory...", file=sys.stdout)
        with torch.no_grad():
            # Create the raw summary vector
            prompt_hidden_state = self.model.embed_tokens(original_input_ids)
            turn_summary_vector = torch.mean(prompt_hidden_state, dim=1)

        # Store the vector and the original input_ids in the memory list
        projected_vector = self.gnn.query_projection(turn_summary_vector)

        # Store the PROJECTED vector and the original input_ids in the memory list
        self.memory_graph.append({
            "vector": projected_vector.cpu(), # Store on CPU to save GPU memory
            "input_ids": original_input_ids.cpu()
        })
        print(f"Added new node. Graph now has {len(self.memory_graph)} nodes.", file=sys.stderr)