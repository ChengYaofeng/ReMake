from collections import OrderedDict

def clean_state_dict(state_dict):
    """Remove 'module.' prefix from keys in a state_dict (for DDP compatibility)"""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "") if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def load_model_weights(model, state_dict):
    try:
        model.load_state_dict(state_dict)
    except RuntimeError:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k = "module." + k
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
