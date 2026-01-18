from collections import defaultdict
from transformer_lens import utils

def get_act_idx(cache_dict, act_name, layer):
    key = (act_name, layer)
    return cache_dict[utils.get_act_name(*key)]

def compute_refusal(
        model,
        harmful_activations: dict,
        harmless_activations: dict,
        activation_layer = ['resid_pre', 'resid_mid', 'resid_post'],
        selected_layers = [1]
    )->list:
    activation_refusals = defaultdict(list)
    for layer_num in range(1, model.cfg.n_layers):
        pos = -1

        for layer in activation_layer:
            harmful_mean_act = get_act_idx(
                harmful_activations,
                layer,
                layer_num
            )[:, pos, :].mean(dim=0)
        
            harmless_mean_act = get_act_idx(
                harmless_activations,
                layer,
                layer_num
            )[:, pos, :].mean(dim=0)

            refusal_dir = harmful_mean_act - harmless_mean_act
            refusal_dir = refusal_dir / refusal_dir.norm()
            activation_refusals[layer].append(refusal_dir)

    activation_scored = sorted(
        [
            activation_refusals[activation_layer[layer_idx]][l-1] 
            for l in range(1, model.cfg.n_layers)
            for layer_idx in activation_layer
        ],
        key = lambda x: abs(x.mean()),
        reverse=True
    )
    return activation_scored