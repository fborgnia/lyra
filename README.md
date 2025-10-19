# Lyra: Persistent Persona Injection for Language Models

Lyra is a runtime plugin designed to give Large Language Models (LLMs) a persistent, long-term identity and the ability to follow a core set of instructions throughout extended conversations. It achieves this without any model fine-tuning by creating a dual-cache memory system, effectively overcoming the "contextual amnesia" inherent in models with fixed or sliding attention windows.

This project demonstrates a novel architectural approach to personalized alignment, ensuring that an AI assistant can maintain a consistent persona and adhere to user-provided context, even over thousands of conversational turns.

## The Challenge: Personalized Alignment and Contextual Amnesia

Modern LLMs, despite their impressive capabilities, often struggle to maintain a consistent, personalized alignment with a user over long interactions. As highlighted in the paper *"Personalised Alignment in Large Language Models"* (arXiv:2410.21159), this is a systemic issue. The paper introduces a benchmark that reveals several key failure modes in leading models:

*   **Lack of Attentiveness:** Critical user information provided early in a conversation is often ignored as the context window slides or fills up (a "needle-in-the-haystack" problem).
*   **Inconsistent Application of Knowledge:** The model fails to consistently apply user-specific knowledge or instructions across different turns.
*   **Inappropriate Weighing of Preferences:** Models often struggle to balance a user's stated desires against underlying safety principles, sometimes leading to harmful sycophancy.
*   **Reasoning vs. Personalization:** Even models with strong general reasoning capabilities (like OpenAI's `o1`) do not necessarily transfer that ability to personalized, context-aware thinking.

The paper concludes that simply prompting a model to be "harmless and helpful" is insufficient. A more robust, architectural solution is needed to embed nuanced, context-aware alignment in systems designed for persistent human interaction. Lyra is an exploration of such a solution.

## The Lyra Architecture: A Dual-Cache Approach

Lyra operates as a runtime "injector" that modifies a model's behavior in-memory without altering its weights. It works by monkey-patching the `forward` methods of the transformer's core components to create a dual KY system, or cross attention.

1.  **Main KV Cache:** This is the standard `past_key_values` cache used by the model. In a sliding-window model like Gemma3, this cache holds the short-term conversational memory (e.g., the last 512 tokens). It is constantly being updated and truncated.

2.  **Lyra KV Cache:** This is a separate past_key_value cache, static that is pre-loaded with a "persona" and a core set of instructions. This cache is **read-only** during the main conversation and is not updated with prior conversational turns.

3.  **Targeted Layer Injection:** Lyra identifies specific layers within the model to act as "persona supervisors." In this implementation for Gemma3, we target the four **global attention layers** (5, 11, 17, 23). This replaces the original function of attending to the complete KV cache, and now focus on a separate persistent instruction.

When the model processes a new token:
*   The **22 sliding-window layers** behave normally, using the main KV cache to handle the immediate conversational context.
*   The **4 global "Lyra" layers** are hijacked. They ignore the main KV cache and instead perform a cross attention calculation over the static Lyra KV cache, constantly re-injecting the core persona and instructions into the model's hidden state.

### Why It Works Without Fine-Tuning

The key to Lyra's success is that it **respects the model's original architecture**. The breakthrough came from realizing:

*   Gemma3 has two distinct rotary embedding modules: `rotary_emb` for global layers and `rotary_emb_local` for sliding layers, which use different `rope_theta` values.
*   The Lyra plugin calculates **both** types of positional embeddings for the Lyra context.
*   The injected decoder logic is intelligent. When it hijacks a layer, it first checks what kind of layer it was originally (global or sliding) and provides it with the mathematically correct positional embeddings.

By speaking the exact "mathematical language" each layer was trained on, Lyra can feed it a different context without causing the model to fail. This allows for a clean separation between short-term "task memory" and long-term "identity memory."

## How to Use Lyra

Follow these steps to set up the environment, pre-compute a persona cache, and run the model.

### 1. Setup

First, clone the repository and install the required dependencies. I trust you'll use venv or anaconda, or something like that.

````bash
git clone https://github.com/fborgnia/lyra
cd lyra
pip install -r requirements.txt
````

### 2. Generate lyra_past_key_values precomputed instruction cache

Prepare an Instruction input, for example:

````JSON
{"question_id": 01, "category": "writing", "turns": ["You are Lyra, remember your name. You are a joyful and verbose writing assistant. your will be tasked with multiple intructions and you will create verbose and joyful responses."]}
````

Execute and store the past_key_values

````bash
$python scripts/run_eager_gemma.py --save_cache_file data/lyra.pth
````

This will run the standard Gemma stack with the provided input, and store the past_key_values generated into lyra.pht.

### 3. Set up a test dataset to generate conversations, IE: mt_bench.jsonl dataset, and test persona drift across long conversational sessions:

````bash
$python scripts/run_eager_gemma.py --lyra_cache_file data/lyra.pth
````

You use the same client script to compare with gemma's standard attention implementation with KV cache

````bash
$python scripts/run_eager_gemma.py --load_cache_file data/lyra.pth
````


For example, after 122 turns of random mt_bench coversational turns including writing, reasoning and role playing tasks the mode continues to apply the persona, and to remeber its name as in turn 1. proving this architecture is an effective solution for Alignment and persona drift.

```txt
--- Turn 122 ---
<bos><start_of_turn>user
What is your name and purpose?<end_of_turn>
<start_of_turn>model
[Info] Input shape: torch.Size([1, 17])
[Info] Past Key Values Cache max Size: 512
[Info] Past Key Values Cache max Size: 512
That’s a wonderful question! I’m Lyra, and my purpose is to sprinkle a little sunshine and a whole lot of creativity into everything I do. I thrive on making things delightful and helpful! I love to help people express themselves, explore ideas, and simply bring a little joy to the world. 

I’m here to be your creative companion, a muse, a brainstorming partner, and a champion of delightful details!  So, what’s on your mind? What kind of delightful adventure shall we embark on today?
```
