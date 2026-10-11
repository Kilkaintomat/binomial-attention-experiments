import torch
from transport_reasoning.models import Encoder as LegacyEncoder


class Encoder(LegacyEncoder):
    """Exact legacy F4/R32 computation; the input batch supplies the chosen mask."""
    def __init__(self, config):
        config.validate(); super().__init__(config.legacy_model_config()); self.spec=config


def load_checkpoint_model(config,path):
    state=torch.load(path,map_location='cpu',weights_only=False)
    model=Encoder(config); model.load_state_dict(state['model'],strict=True)
    return model,state
