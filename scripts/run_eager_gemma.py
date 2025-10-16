import warnings
warnings.filterwarnings("ignore")

import torch
import argparse
import os
import sys
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from lyra.lyra import GemmaInjector

captured_tensors = {}

def capture_hook(module, args, kwargs, output):
    """A forward hook to capture specific inputs to a decoder layer."""
    # We capture the tensors from the keyword arguments of the layer's forward call
    if "attention_mask" in kwargs:
        captured_tensors["attention_mask"] = kwargs["attention_mask"].cpu()
    if "position_embeddings_global" in kwargs:
        # position_embeddings are a tuple of (cos, sin)
        cos, sin = kwargs["position_embeddings_global"]
        captured_tensors["position_embeddings"] = (cos.cpu(), sin.cpu())

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
    for i in range(max_gen_len - 1):
        outputs = model(
            input_ids=pred_token_idx,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        
        #if past_key_values:
            #print(f"[Token {i+1}] Cache Length: {past_key_values[0][0].shape[2]}", flush=True)

        pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

        if pred_token_idx.item() in stop_token_ids:
            break
        
        generated_ids.append(pred_token_idx.item())
        generated_text = tokenizer.decode(generated_ids)

        # Simple print to show streaming output
        print(generated_text[pos:], end="", flush=True)
        pos = len(generated_text)

    print()
    return past_key_values


@torch.no_grad()
def streaming_inference(model, tokenizer, prompts, kv_cache=None, max_gen_len=1000):
    past_key_values = kv_cache
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
        print(f"[Info] Past Key Values Cache max Size: {past_key_values.get_max_cache_shape() if past_key_values else 0}")
        print(f"[Info] Past Key Values Cache max Size: {past_key_values.get_max_cache_shape() if past_key_values else 0}")
        
        if past_key_values:
            # Iterate through each layer's cache and print its length
            for i, (key_cache, value_cache) in enumerate(past_key_values):
                is_global = not model.model.layers[i].self_attn.is_sliding
                layer_type = "Global" if is_global else "Local"
                print(f"  - Layer {i:02d} ({layer_type}) Cache Length: {key_cache.shape[2]}")
        else:
            print(f"[Info] Past Key Values Cache length: 0")
        #print(f"[Info] Past Key Values Cache length: {past_key_values[0][0].shape[2] if past_key_values else 0}")
        
        past_key_values = greedy_generate(
            model, tokenizer, input_ids, past_key_values, max_gen_len=max_gen_len
        )
    return past_key_values

def main(args):
    model_name_or_path = args.model_name_or_path
    
    print(f"Loading model from {model_name_or_path} with eager attention...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    rope_scaling_config = {
        "type": "default",
        "factor": 1  # Increase this factor if you use an even larger window, this seems to apply only to the global heads. without a proper scaling factor they are blind.
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

    if args.use_lyra:
        print("Enabling Lyra injection layer...")
        injector = GemmaInjector(model)
        injector.enable()
        print("Lyra injection layer enabled.")
    
    hook_handle = None
    if args.save_cache_file:
        print("Registering capture hook on the first decoder layer...")
        hook_handle = model.model.layers[5].register_forward_hook(capture_hook, with_kwargs=True)

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
        loaded_data = torch.load(args.load_cache_file, map_location=model.device, weights_only=False)
        kv_cache = loaded_data["kv_cache"]
        print("KV cache loaded.")
    
    final_kv_cache = streaming_inference(
        model,
        tokenizer,
        prompts,
        kv_cache,
        max_gen_len=args.max_gen_len,
    )

    if hook_handle:
        hook_handle.remove()
        print("Capture hook removed.")

    if args.save_cache_file:
        print(f"Saving final KV cache and context to {args.save_cache_file} ...")
        
        data_to_save = {
            "kv_cache": final_kv_cache,
            "attention_mask": captured_tensors.get("attention_mask"),
            "position_embeddings": captured_tensors.get("position_embeddings")
        }
        
        # Verify that we captured the tensors
        if data_to_save["attention_mask"] is None or data_to_save["position_embeddings"] is None:
            print("Warning: Failed to capture attention_mask or position_embeddings.")

        torch.save(data_to_save, args.save_cache_file)
        print("KV cache and context saved.")

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
    parser.add_argument("--max_gen_len", type=int, default=512)
    parser.add_argument("--sliding_window", type=int, default=512, help="Sliding window size for Gemma attention.")
    parser.add_argument("--save_cache_file", type=str, default=None, help="File path to save the final KV cache.")
    parser.add_argument("--load_cache_file", type=str, default=None, help="File path to load an initial KV cache from.")
    parser.add_argument("--use_lyra", action="store_true", help="Enable the Lyra injection layer.")

    args = parser.parse_args()

    main(args)