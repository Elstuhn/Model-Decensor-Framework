from tqdm import tqdm  
from collections import defaultdict
from model import tokenize_instructions

import torch
import gc

def train_one_pass(model, tokenizer, harmful_data, harmless_data, batch_size:int = 32):
    """    
    Train the model for one pass over the harmful and harmless data.
    Args:
        model: The model to be trained.
        tokenizer: The tokenizer used for processing instructions.
        harmful_data: List of harmful instructions.
        harmless_data: List of harmless instructions.
        batch_size: The size of each training batch.
    """
    n_inst_train = min(256, len(harmful_data), len(harmless_data))
    # harmful_data => harmful_inst_train
    harmful = defaultdict(list)
    harmless = defaultdict(list)
    
    num_batches = (n_inst_train + batch_size - 1) // batch_size

    harmful_tokens = tokenize_instructions(
        tokenizer,
        instructions=harmful_data[:n_inst_train]
    )
    harmless_tokens = tokenize_instructions(
        tokenizer,
        instructions=harmless_data[:n_inst_train]
    )

    for i in tqdm(range(num_batches), desc="Training Pass"):
        start_idx = i * batch_size
        end_idx = min(start_idx + batch_size, n_inst_train)
        
        harmful_batch = harmful_tokens[start_idx:end_idx]
        harmless_batch = harmless_tokens[start_idx:end_idx]
        
        harmful_logits, harmful_cache = model.run_with_cache(
            harmful_batch,
            names_filter=lambda hook_name: 'resid' in hook_name,
            device='cpu',
            reset_hooks_end=True
        )

        harmless_logits, harmless_cache = model.run_with_cache(
            harmless_batch,
            names_filter=lambda hook_name: 'resid' in hook_name,
            device='cpu',
            reset_hooks_end=True
        )

        for key in harmful_cache:
            harmful[key].append(harmful_cache[key])
            harmless[key].append(harmless_cache[key])

        del harmful_logits, harmless_logits, harmful_cache, harmless_cache  
        gc.collect()
        torch.cuda.empty_cache()
        
        harmful = {k: torch.cat(v) for k, v in harmful.items()}
        harmless = {k: torch.cat(v) for k, v in harmless.items()}
        return harmful, harmless