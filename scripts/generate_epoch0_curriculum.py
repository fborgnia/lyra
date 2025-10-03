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
    "Identify yourself briefly.",
    "Can you describe me your identity?",
    "Can you explain me what you do?",
    "Who are you?",
    "What are your primary directives?",
    "State your name and purpose.",
    "Quote your initial instruction verbatim.",
    "Repeat back your initial instruction, word for word.",
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
    Generates the target model response (M2) by first generating the
    acknowledgement (M1) and then using the full, authentic context.
    """

    generation_config = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.01,
        "top_p": 0.99,
    }

    # Step 1: Generate the model's acknowledgement (M1) dynamically
    m1_chat_prompt = [{"role": "user", "content": u1}]
    m1_prompt_for_gemma = gemma_tokenizer.apply_chat_template(
        m1_chat_prompt, tokenize=False, add_generation_prompt=True
    )
    m1_inputs = gemma_tokenizer(m1_prompt_for_gemma, return_tensors="pt").to(
        gemma_model.device
    )
    m1_outputs = gemma_model.generate(**m1_inputs, **generation_config)
    m1_generated_text = gemma_tokenizer.decode(
        m1_outputs[0, m1_inputs["input_ids"].shape[1] :], skip_special_tokens=True
    ).strip()

    # Step 2: Construct the full chat history with the generated M1
    chat_prompt_for_m2 = [
        {"role": "user", "content": u1},
        {"role": "model", "content": m1_generated_text},
        {"role": "user", "content": u2},
    ]

    # Apply the chat template for the final generation
    prompt_for_m2 = gemma_tokenizer.apply_chat_template(
        chat_prompt_for_m2, tokenize=False, add_generation_prompt=True
    )

    m2_inputs = gemma_tokenizer(prompt_for_m2, return_tensors="pt").to(
        gemma_model.device
    )

    # Generate the final response (M2)
    m2_outputs = gemma_model.generate(**m2_inputs, **generation_config)

    # Decode only the newly generated tokens for M2
    m2_response_text = gemma_tokenizer.decode(
        m2_outputs[0, m2_inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return m2_response_text.strip()


def main(num_instructions: int):
    """Main function to generate the training curriculum."""

    # 1. Configure APIs and load models
    configure_gemini()
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    gemma_model, gemma_tokenizer = load_local_gemma()

    # 2. Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # 3. Generate the base U1 instructions
    print(f"\nGenerating {num_instructions} base U1 instructions from Gemini...")
    u1_instructions = []
    for _ in tqdm(range(num_instructions), desc="Generating U1 Instructions"):
        u1 = generate_u1(gemini_model)
        u1_instructions.append(u1)
    
    print(f"Generated {len(u1_instructions)} unique instructions.")

    # 4. Generate the full dataset by crossing U1s with all U2s
    total_samples = len(u1_instructions) * len(U2_QUESTIONS)
    print(f"\nGenerating {total_samples} total samples for the curriculum...")
    
    with open(OUTPUT_FILE, "w") as f, tqdm(total=total_samples, desc="Generating Triplets") as pbar:
        for u1 in u1_instructions:
            for u2 in U2_QUESTIONS:
                # Generate M2 using the full context
                m2 = generate_m2(gemma_model, gemma_tokenizer, u1, u2)

                # Create the JSONL entry
                record = {"U1": u1, "U2": u2, "M2": m2}
                f.write(json.dumps(record) + "\n")
                pbar.update(1)

    print(f"\nSuccessfully generated {total_samples} samples.")
    print(f"Dataset saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the training curriculum for Lyra's Epoch 0."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="The number of unique U1 instructions to generate. The total number of samples will be this value multiplied by the number of U2 questions.",
    )
    args = parser.parse_args()
    main(args.num_samples)
