Plan: Aligning Memory Operations at Layer 8 via Hooks
The core idea is to register two hooks on layer 8 of the Gemma model:

A pre-forward hook to inject memory before the layer executes.
A forward hook to capture the layer's output for memory generation.
Phase 1: Setup Hooks and State Management in __init__
We'll configure everything in the __init__ method of GemmaWithMemory.

Step 1.1: Define Injection Layer and State

Add self.injection_layer_idx = 8.
Add a state flag self.is_prompt_processing = False. This is crucial to ensure memory is injected only once per generate call (during prompt processing), not for every single generated token.
Step 1.2: Implement the Injection Pre-Hook

We'll create a method _injection_pre_hook that will be registered as a pre-forward hook on layer 8.
This hook will check if self.is_prompt_processing. If true, it will:
Take the incoming hidden_states.
Create a query_vector by pooling these states.
Call the GNN to retrieve memory.
Prepend the memory to the hidden_states and its attention_mask.
Return the modified hidden_states and attention_mask.
Set self.is_prompt_processing = False so it doesn't run again in the same generate loop.
Step 1.3: Implement the Memory Capture Hook

We'll create a simple _memory_capture_hook method.
It will be registered as a forward hook on layer 8.
Its only job is to grab the output of layer 8 and store it in a class variable, like self.last_layer_8_output, so _update_memory can use it later.
Phase 2: Adapt generate() and _update_memory()
The generate and _update_memory methods become much simpler with this design.

Step 2.1: Simplify generate()

The generate method's main job is now to manage the state for the hooks.
Before calling self.gemma.generate(), it will set self.is_prompt_processing = True.
The call to self.gemma.generate() will now be very clean, as the hooks handle all the complex intervention.
Step 2.2: Simplify _update_memory()

This method no longer needs to receive hidden states as an argument.
It will simply use the self.last_layer_8_output that was captured by the forward hook.
This hook-based plan is far superior. It avoids rewriting library code, is less likely to break with updates, and clearly separates our custom logic from the base model's operations.