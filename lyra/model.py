import os
from transformers import Gemma3ForCausalLM, AutoTokenizer

class Lyra(Gemma3ForCausalLM):
    """
    Lyra is a Gemma 3 model. This version is a baseline implementation that
    adheres to the standard Gemma3ForCausalLM architecture.
    """
    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DEFAULT_MODEL_PATH = os.path.join(_project_root, 'models/gemma-3-1b-it')

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path=None, *model_args, **kwargs):
        if pretrained_model_name_or_path is None:
            pretrained_model_name_or_path = cls.DEFAULT_MODEL_PATH
        
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        model.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        return model
