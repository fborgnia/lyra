from transformers import Gemma3ForCausalLM, Gemma3Config

class Lyra(Gemma3ForCausalLM):
    """
    Lyra is a Gemma 3 model. This version is a baseline implementation that
    adheres to the standard Gemma3ForCausalLM architecture.
    """
    def __init__(self, config: Gemma3Config):
        super().__init__(config)

    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs):
        """
        Loads a pretrained Gemma 3 model and wraps it in the Lyra class.
        """
        print(f"Loading base model from {model_path}...")
        base_model = Gemma3ForCausalLM.from_pretrained(model_path, **kwargs)
        
        # Initialize Lyra with the same config
        lyra_model = cls(base_model.config)
        
        # Load the state dict from the pretrained model
        lyra_model.load_state_dict(base_model.state_dict())
        
        print("Model loaded successfully into Lyra container.")
        return lyra_model
