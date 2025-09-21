import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import torch
from lyra.model import GemmaWithMemory

@pytest.fixture(scope="module")
def model():
    """Fixture to load the local Gemma 3 model once for all tests."""
    # This will use the default path './models/gemma-3-1b-it' defined in the model's __init__
    return GemmaWithMemory()

def test_model_instantiation(model):
    """Tests if the GemmaWithMemory model can be instantiated correctly."""
    assert model is not None
    assert model.gemma is not None
    assert model.tokenizer is not None
    assert model.memory_graph is None
    assert model.start_of_turn_token_id is not None
    assert model.end_of_turn_token_id is not None

def test_forward_pass_triggers_memory_reset(model, capsys):
    """
    Tests if the forward() method correctly detects the start of a turn
    and prints the memory reset message.
    """
    # Simulate a new conversation by including the <start_of_turn> token
    prompt = "<start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n<start_of_turn>model\n"
    inputs = model.tokenizer(prompt, return_tensors="pt")

    # The forward pass should detect the start token and reset the memory
    model.forward(**inputs)

    captured = capsys.readouterr()
    assert "New conversation detected. Resetting memory." in captured.out

def test_generate_triggers_memory_update_placeholder(model, capsys):
    """
    Tests if the generate() method correctly triggers the placeholder
    for the memory update logic after generating a response.
    """
    prompt = "<start_of_turn>user\nWhat is the capital of France?<end_of_turn>\n<start_of_turn>model\nIt's Paris.<end_of_turn>\n<start_of_turn>user\nIs it pretty?<end_of_turn>\n<start_of_turn>model"
    inputs = model.tokenizer(prompt, return_tensors="pt")
    
    # Generate a short response. The model should produce an <end_of_turn> token.
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=15,
        eos_token_id=model.end_of_turn_token_id
    )

     # Decode the full output to text for visual inspection
    full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=False)
    captured = capsys.readouterr()
    
    with capsys.disabled():
        print(f"\n--- Model Output ---\n{full_text}\n--------------------")
        print(f"--- Captured STDOUT ---\n{captured.out.strip()}\n---------------------")

    #captured = capsys.readouterr()
    # Check that the placeholder message for the memory update is printed
    assert "Extracted hidden states for the turn with shape" in captured.out