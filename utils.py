import copy
import torch.nn as nn

def aggregate_models(w, client_state_dict, alpha=0.1):
    result = copy.deepcopy(w)
    for key in result.keys():
        result[key] =  w[key] * (1 - alpha) + client_state_dict[key] * alpha
    return result

def get_loss_function(model):
    if isinstance(model, nn.Module):
        last_layer = list(model.children())[-1]
        if isinstance(last_layer, nn.Softmax):
            return nn.NLLLoss()
        else:
            return nn.CrossEntropyLoss()
    else:
        raise ValueError("Invalid model type.")
