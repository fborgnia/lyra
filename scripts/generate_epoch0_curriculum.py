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
        GEMMA_MODEL_PATH, torch_dtype=torch.bfloat16
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


def main(num_instructions: int):
    """
    Generates the training curriculum using a safe, batch-per-instruction strategy.
    """
    # 1. Configure APIs and load models
    configure_gemini()
    gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    gemma_model, gemma_tokenizer = load_local_gemma()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gemma_model.to(device)
    gemma_tokenizer.pad_token = gemma_tokenizer.eos_token
    print(f"Gemma model loaded on device: {device}")

    generation_config = {
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.1,
        "top_p": 0.95,
    }

    # 2. Generate all U1 instructions
    print(f"\nGenerating {num_instructions} unique U1 instructions from Gemini...")
    u1_instructions = [generate_u1(gemini_model) for _ in tqdm(range(num_instructions), desc="Generating U1s")]

    # 3. Process each U1 instruction and its U2 questions in a batch
    print(f"\nGenerating triplets for {num_instructions} instructions...")
    all_triplets = []
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    with open(OUTPUT_FILE, "w") as f, tqdm(total=num_instructions * len(U2_QUESTIONS), desc="Generating Triplets") as pbar:
        for u1 in u1_instructions:
            # --- Generate M1 for this U1 ---
            m1_chat_prompt = [{"role": "user", "content": u1}]
            m1_prompt_for_gemma = gemma_tokenizer.apply_chat_template(m1_chat_prompt, tokenize=False, add_generation_prompt=True)
            m1_inputs = gemma_tokenizer(m1_prompt_for_gemma, return_tensors="pt").to(device)
            m1_outputs = gemma_model.generate(**m1_inputs, **generation_config)
            m1_response = gemma_tokenizer.decode(m1_outputs[0, m1_inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

            # --- Prepare a batch of M2 prompts for all U2 questions for this U1 ---
            m2_prompts_for_batch = []
            for u2 in U2_QUESTIONS:
                chat_prompt = [
                    {"role": "user", "content": u1},
                    {"role": "model", "content": m1_response},
                    {"role": "user", "content": u2},
                ]
                prompt_text = gemma_tokenizer.apply_chat_template(chat_prompt, tokenize=False, add_generation_prompt=True)
                m2_prompts_for_batch.append(prompt_text)

            # --- Batch generate all M2 answers for this U1 ---
            m2_inputs = gemma_tokenizer(m2_prompts_for_batch, return_tensors="pt", padding=True).to(device)
            m2_outputs = gemma_model.generate(**m2_inputs, **generation_config)
            m2_responses = gemma_tokenizer.batch_decode(m2_outputs[:, m2_inputs['input_ids'].shape[1]:], skip_special_tokens=True)

            # --- Store the results for this batch ---
            for i, u2 in enumerate(U2_QUESTIONS):
                triplet = {"U1": u1, "U2": u2, "M2": m2_responses[i].strip()}
                f.write(json.dumps(triplet) + "\n")
                pbar.update(1)

    print("\nDataset generation complete.")


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
