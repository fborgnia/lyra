from transformers import Gemma3ForCausalLM, AutoTokenizer

class Lyra(Gemma3ForCausalLM):
    """
    Lyra is a Gemma 3 model. This version is a baseline implementation that
    adheres to the standard Gemma3ForCausalLM architecture.
    """
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        return model
