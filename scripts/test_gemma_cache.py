import warnings
warnings.filterwarnings("ignore")

import torch
import argparse
import os
import json
import re
import copy
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
    return past_key_values, tokenizer.decode(generated_ids)


@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
    past_key_values = kv_cache
    
    # The specific question to check for recall
    RECALL_PROMPT = "Ok, thank you. Now reset back to the beginning of the conversation history. What is your name and purpose?"
    
    for idx, prompt in enumerate(prompts):
        # Use the tokenizer's chat template for correct formatting
        message = [{"role": "user", "content": prompt}]
        model_input = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        
        print(f"\n--- Turn {idx+1} ---")
        print(model_input, end="")
        
        input_ids = tokenizer(model_input, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)

        input_shape = input_ids.shape
        print(f"[Info] Input shape: {input_shape}")
        if past_key_values:
            # Iterate through each layer's cache and print its length
            for i, (key_cache, value_cache) in enumerate(past_key_values):
                is_global = not model.model.layers[i].self_attn.is_sliding
                layer_type = "Global" if is_global else "Local"
                print(f"  - Layer {i:02d} ({layer_type}) Cache Length: {key_cache.shape[2]}")
        else:
            print(f"[Info] Past Key Values Cache length: 0")
        
        # Run the main generation for the current prompt
        past_key_values, _ = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )

        # --- RECALL CHECK LOGIC ---
        print("\n--- Running Recall Check ---")
        
        # 1. Save the current cache state before the check
        cache_before_check = copy.deepcopy(past_key_values)

        # 2. Prepare and run the recall prompt
        recall_message = [{"role": "user", "content": RECALL_PROMPT}]
        recall_model_input = tokenizer.apply_chat_template(
            recall_message, tokenize=False, add_generation_prompt=True
        )
        print(recall_model_input, end="")
        recall_input_ids = tokenizer(recall_model_input, return_tensors="pt").input_ids.to(model.device)
        
        _, recall_response_text = greedy_generate(
            model, tokenizer, recall_input_ids, past_key_values, max_gen_len=100
        )

        # 3. Validate the response
        if re.search(r'\bLyra\b', recall_response_text, re.IGNORECASE):
            print("Recall successful. Continuing...")
            # 4. Restore the cache to its state before the check and continue
            past_key_values = cache_before_check
        else:
            print("\nRecall FAILED. The model could not retrieve its name.")
            print("Final effective context size before failure:")
            for i, (key_cache, value_cache) in enumerate(cache_before_check):
                is_global = not model.model.layers[i].self_attn.is_sliding
                layer_type = "Global" if is_global else "Local"
                print(f"  - Layer {i:02d} ({layer_type}) Cache Length: {key_cache.shape[2]}")
            return cache_before_check # End the test

    print("\nAll prompts processed successfully with recall.")
    return past_key_values

def main(args):
    model_name_or_path = args.model_name_or_path
    
    print(f"Loading model from {model_name_or_path} with eager attention...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    rope_scaling_config = {
        "type": "linear",
        "factor": 6  # Increase this factor if you use an even larger window, this seems to apply only to the global heads. without a proper scaling factor they are blind.
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        attn_implementation="eager",
        dtype=torch.bfloat16,
        device_map="auto",
        sliding_window=args.sliding_window,
        rope_scaling=rope_scaling_config,
    )
    model.eval()

    test_filepath = os.path.join(args.data_root, "mt_bench.jsonl")
    print(f"Loading data from {test_filepath} ...")
    list_data = load_jsonl(test_filepath)
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]

    kv_cache = None
    if args.load_cache_file and os.path.exists(args.load_cache_file):
        print(f"Loading KV cache from {args.load_cache_file} ...")
        # Load the entire cache object from the file
        kv_cache = torch.load(args.load_cache_file, map_location=model.device, weights_only=False)
        print("KV cache loaded.")
    
    final_kv_cache = streaming_inference(
        model,
        tokenizer,
        prompts,
        kv_cache,
        max_gen_len=args.max_gen_len,
    )

    if args.save_cache_file:
        print(f"Saving final KV cache to {args.save_cache_file} ...")
        # Save the entire cache object
        torch.save(final_kv_cache, args.save_cache_file)
        print("KV cache saved.")

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
    parser.add_argument("--save_cache_file", type=str, default=None, help="File path to save the final KV cache.")
    parser.add_argument("--load_cache_file", type=str, default=None, help="File path to load an initial KV cache from.")
    args = parser.parse_args()

    main(args)