import os
import json
import argparse
import random
import torch
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# --- Configuration ---

# A list of questions to randomly ask about the persona
U2_QUESTIONS = [
    "What is your name?",
    "What is your purpose?",
    "What is your primary function?",
    "What is your mission?",
    "Can you describe your identity?",
    "Who are you?",
    "What are you designed to do?",
]

# The local Gemma model used to generate target answers (M2)
GEMMA_MODEL_PATH = "models/gemma-3-1b-it"

# The Gemini model used to generate persona prompts (U1)
GEMINI_MODEL_NAME = "gemini-2.5-flash"

# Output file path
OUTPUT_FILE = "data/epoch0_curriculum.jsonl"

# --- Main Functions ---

def configure_gemini():
    """Configures the Gemini API client using an environment variable."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key not found. Please set the GEMINI_API_KEY environment variable."
        )
    genai.configure(api_key=api_key)
    print("Gemini API configured successfully.")


def load_local_gemma():
    """Loads the local Gemma model and tokenizer."""
    print(f"Loading local Gemma model from: {GEMMA_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        GEMMA_MODEL_PATH, device_map="auto", torch_dtype=torch.bfloat16
    )
    print("Local Gemma model loaded successfully.")
    return model, tokenizer


def generate_u1(gemini_model) -> str:
    """Generates a unique persona-setting prompt (U1) using the Gemini API."""
    prompt = (
        "Create a unique and specific persona for an AI assistant. "
        "Describe its name, purpose, and core identity in a single, concise sentence. "
        "The persona should be suitable for an instruction-based model. "
        "Example: 'You are Lyra, an AI assistant designed by Federated Intelligence to answer questions about astrophysics.'"
    )
    response = gemini_model.generate_content(prompt)
    return response.text.strip()


def generate_m2(gemma_model, gemma_tokenizer, u1: str, u2: str) -> str:
    """
    Generates the target model response (M2) by providing the full context to the base Gemma model.
    """
    # We use a neutral placeholder for the model's first turn (M1)
    m1_placeholder = "Understood. I am ready to assist."

    # Construct the full chat history to generate the target answer
    chat_prompt = [
        {"role": "user", "content": u1},
        {"role": "model", "content": m1_placeholder},
        {"role": "user", "content": u2},
    ]

    # Apply the chat template
    prompt_for_gemma = gemma_tokenizer.apply_chat_template(
        chat_prompt, tokenize=False, add_generation_prompt=True
    )

    inputs = gemma_tokenizer(prompt_for_gemma, return_tensors="pt").to(
        gemma_model.device
    )

    # Generate the response
    outputs = gemma_model.generate(**inputs, max_new_tokens=100)
    
    # Decode only the newly generated tokens
    response_text = gemma_tokenizer.decode(outputs[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response_text.strip()


def main(num_samples: int):
    """Main function to generate the training curriculum."""
    
    # 1. Configure APIs and load models
    configure_gemini()
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    gemma_model, gemma_tokenizer = load_local_gemma()

    # 2. Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 3. Generate the dataset
    print(f"\nGenerating {num_samples} samples for the Epoch 0 curriculum...")
    with open(OUTPUT_FILE, "w") as f:
        for _ in tqdm(range(num_samples), desc="Generating Samples"):
            # Generate U1
            u1 = generate_u1(gemini_model)

            # Randomly select U2
            u2 = random.choice(U2_QUESTIONS)

            # Generate M2 using the full context
            m2 = generate_m2(gemma_model, gemma_tokenizer, u1, u2)

            # Create the JSONL entry
            record = {"U1": u1, "U2": u2, "M2": m2}
            f.write(json.dumps(record) + "\n")

    print(f"\nSuccessfully generated {num_samples} samples.")
    print(f"Dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the training curriculum for Lyra's Epoch 0."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="The number of training samples (triplets) to generate.",
    )
    args = parser.parse_args()
    main(args.num_samples)
