import sys
import os
import pytest
import torch

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lyra.model import Lyra

@pytest.fixture(scope="module")
def model():
    """Fixture to load the Lyra model once for all tests."""
    # This will use the default path './models/gemma-3-1b-it' defined in the model's __init__
    return Lyra()

def test_model_instantiation(model):
    """Tests if the Lyra model can be instantiated correctly."""
    assert model is not None
    assert model.memory_store is not None
    assert model.memory_archival_block is not None
    assert model.tokenizer is not None
    assert hasattr(model.model.layers[0], 'memory_injection_block'), "Decoder layers should have a memory_injection_block"
    assert hasattr(model.model.layers[0], 'post_memory_layernorm'), "Decoder layers should have a post_memory_layernorm"

def test_generate_and_archive(model, capsys):
    """
    Tests if the generate() method runs, triggers memory archival,
    and that the memory store is populated.
    """
    # Clear any existing memories
    model.memory_store.memories.clear()
    assert len(model.memory_store.retrieve_all()) == 0

    prompt = "What is the capital of France?"
    inputs = model.tokenizer(prompt, return_tensors="pt")
    
    # Generate a short response
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=5,
    )

    # Decode the full output to text for visual inspection
    full_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    with capsys.disabled():
        print(f"\n--- Model Output ---\n{full_text}\n--------------------")

    captured = capsys.readouterr()
    
    # 1. Check that the archival block was called
    assert "--- Memory Archival Block ---" in captured.out
    assert "Created index vector shape" in captured.out

    # 2. Check that a memory was actually added to the store
    memories = model.memory_store.retrieve_all()
    assert len(memories) == 1
    
    # 3. Check the contents of the stored memory package
    hidden_state, attention_mask, index_vector = memories[0]
    assert isinstance(hidden_state, torch.Tensor)
    assert isinstance(attention_mask, torch.Tensor)
    assert isinstance(index_vector, torch.Tensor)
    assert hidden_state.dim() == 3  # (batch_size, seq_len, hidden_dim)
    assert index_vector.dim() == 2 # (batch_size, hidden_dim)

def test_injection_block_is_called(model, capsys):
    """
    Tests that the memory injection block is called during a generation pass
    when memories are present in the store.
    """
    # Ensure there is a memory in the store from the previous test
    if not model.memory_store.retrieve_all():
        # If the store is empty, run the previous test to populate it
        test_generate_and_archive(model, capsys)

    assert len(model.memory_store.retrieve_all()) > 0

    prompt = "What is its primary language?"
    inputs = model.tokenizer(prompt, return_tensors="pt")
    
    # Generate a response
    model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=5,
    )

    captured = capsys.readouterr()

    # Check that the injection block's placeholder message was printed
    assert "--- Memory Injection Block ---" in captured.out
    assert "Selected 1 memories for injection." in captured.out

def test_injection_selects_first_and_last(model, capsys):
    """
    Tests that the memory injection block selects both the first and last
    memories when more than one is available.
    """
    # Ensure the store has at least two memories.
    # The first one is from the previous test.
    assert model.memory_store.count() >= 1
    
    # Generate another response to add a second memory
    prompt = "And what is its population?"
    inputs = model.tokenizer(prompt, return_tensors="pt")
    model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=5,
    )
    
    # Now there should be at least two memories
    assert model.memory_store.count() >= 2
    
    # Run one more generation to trigger the injection logic
    prompt = "Is it a good place to visit?"
    inputs = model.tokenizer(prompt, return_tensors="pt")
    model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=5,
    )

    captured = capsys.readouterr()

    # Check that the injection block selected two memories
    assert "--- Memory Injection Block ---" in captured.out
    assert "Selected 2 memories for injection." in captured.out