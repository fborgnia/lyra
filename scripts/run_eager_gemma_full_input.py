import warnings
warnings.filterwarnings("ignore")

import torch
import argparse
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

@torch.no_grad()
def greedy_generate(model, tokenizer, input_ids, past_key_values, max_gen_len):
    # Define stop tokens for Gemma
    stop_token_ids = {
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<end_of_turn>"),
    }

    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        # Enable use_cache to utilize past_key_values for faster generation
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    
    generated_ids = [pred_token_idx.item()]

    pos = 0
    for _ in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            # Enable use_cache to utilize past_key_values for faster generation
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

        if pred_token_idx.item() in stop_token_ids:
            break
        
        generated_ids.append(pred_token_idx.item())
        generated_text = tokenizer.decode(generated_ids)

        # Simple print to show streaming output
        print(generated_text[pos:], end="", flush=True)
        pos = len(generated_text)

    print()
    # Return the full generated text along with the cache
    return tokenizer.decode(generated_ids), past_key_values


@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1024):
    # This list will store the entire conversation history.
    conversation_history = []
    for idx, prompt in enumerate(prompts):
        # Add the new user prompt to the history
        conversation_history.append({"role": "user", "content": prompt})

        # Apply the chat template to the entire conversation history
        model_input = tokenizer.apply_chat_template(
            conversation_history, tokenize=False, add_generation_prompt=True
        )
        
        print(f"\n--- Turn {idx+1} ---")
        print(model_input, end="")
        
        input_ids = tokenizer(model_input, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        
        input_shape = input_ids.shape
        print(f"[Info] Input shape: {input_shape}")
        # The past_key_values are not carried over, so we print 0
        #print(f"[Info] Past Key Values Cache max Size: {past_key_values.get_max_cache_shape() if past_key_values else 0}")
        #print(f"[Info] Past Key Values Cache length: {past_key_values[0][0].shape[2] if past_key_values else 0}")
        
        #Pass an empty cache to force the model to one-shot the full input
        generated_text, _ = greedy_generate(
            model, tokenizer, input_ids, None, max_gen_len=max_gen_len
        )

        # Add the model's response to the history for the next turn
        conversation_history.append({"role": "model", "content": generated_text})


def main(args):
    model_name_or_path = args.model_name_or_path
    
    print(f"Loading model from {model_name_or_path} with eager attention...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    rope_scaling_config = {
        "type": "linear",
        "factor": 2  # Increase this factor if you use an even larger window
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        sliding_window=args.sliding_window,
        #rope_scaling=rope_scaling_config,
    )
    model.eval()

    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")
    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    kv_cache = None
    
    streaming_inference(
        model,
        tokenizer,
        prompts,
        kv_cache,
        max_gen_len=args.max_gen_len,
    )

def load_jsonl(file_path):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="models/gemma-3-1b-it"
    )
    parser.add_argument("--data_root", type=str, default="data/")
    parser.add_argument("--max_gen_len", type=int, default=1024)
    parser.add_argument("--sliding_window", type=int, default=512, help="Sliding window size for Gemma attention.")
    args = parser.parse_args()

    main(args)