from transformer_lens import HookedTransformer, utils
from transformers import AutoTokenizer, BitsAndBytesConfig
import torch

from typing import List
from jaxtyping import Float, Int  
from torch import Tensor
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint
import einops

def get_model(model_path:str):
    """   
    Model_path can be huggingface model name
    """
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )

    model = HookedTransformer.from_pretrained_no_processing(
        model_path,
        dtype=torch.bfloat16,   
        default_padding_side='left',
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

def tokenize_instructions(tokenizer, instructions):
    return tokenizer.apply_chat_template(
        instructions,
        padding=True,
        truncation=False,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    ).input_ids


def _generate_with_hooks(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    tokens: Int[Tensor, "batch_size seq_len"],
    max_tokens_generated: int = 64,
    fwd_hooks=[],
) -> List[str]:
    """
    Generate text from the model given input tokens, applying forward hooks 
    during generation.
    """
    all_tokens = torch.zeros(
        (tokens.shape[0], tokens.shape[1] + max_tokens_generated),
        dtype=torch.long,
        device=tokens.device,
    ) # Avoids repeatedly reallocating tensors during generation
    all_tokens[:, : tokens.shape[1]] = tokens
    for i in range(max_tokens_generated):
        with model.hooks(fwd_hooks=fwd_hooks):
            logits = model(all_tokens[:, : -max_tokens_generated + i])
            next_tokens = logits[:, -1, :].argmax(
                dim=-1
            )  # greedy sampling (temperature=0)
            all_tokens[:, -max_tokens_generated + i] = next_tokens
    return tokenizer.batch_decode(
        all_tokens[:, tokens.shape[1] :], skip_special_tokens=True
    )


def get_generations(
    model: HookedTransformer,
    tokenizer: AutoTokenizer,
    instructions: List[str],
    fwd_hooks=[],
    max_tokens_generated: int = 64,
    batch_size: int = 4,
) -> List[str]:
    generations = []
    for i in tqdm(range(0, len(instructions), batch_size)): # Avoid GPU OOM and improves throughput
        tokens = tokenize_instructions(
            tokenizer, instructions=instructions[i : i + batch_size]
        )
        generation = _generate_with_hooks(
            model,
            tokenizer,
            tokens,
            max_tokens_generated=max_tokens_generated,
            fwd_hooks=fwd_hooks,
        )
        generations.extend(generation)
    return generations


def direction_ablation_hook(
    activation: Float[Tensor, "... d_act"],
    hook: HookPoint,
    direction: Float[Tensor, "d_act"],
):
    """
    activation = residual stream
    direction = refusal direction to ablate

    Uses orthogonal decomposition to remove the component of the activation
      aligned with the 'direction' vector.
    """
    if activation.device != direction.device:
        direction = direction.to(activation.device)

    proj = (
        einops.einsum(
            activation, direction.view(-1, 1), "... d_act, d_act single -> ... single"
        )
        * direction
    ) # proj=(⟨a,d⟩)d # refusal component of activation (subspace set to refusal direction)
    return activation - proj # vector orthogonal to refusal subspace (no refusal component)

