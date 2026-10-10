import torch
from transport_reasoning.models import Encoder as LegacyEncoder


class Encoder(LegacyEncoder):
    """Unchanged block and named initialization, independent architecture/filter dispatch."""
    def __init__(self, config):
        config.validate()
        super().__init__(config.legacy_model_config())
        self.spec = config

    def forward(self, batch, return_states=False):
        # Only input features and the task mask enter the computation.
        x = self.embedding(batch['x'])
        states = []
        if self.spec.architecture == 'recurrent':
            for _ in range(self.spec.recurrent_steps):
                x = self.blocks[0](x, batch['mask'])
                if return_states: states.append(x)
        else:
            for block in self.blocks:
                x = block(x, batch['mask'])
                if return_states: states.append(x)
        logits = self.readout(self.norm(x))
        return (logits, states) if return_states else logits


def load_checkpoint_model(config, path):
    state = torch.load(path, map_location='cpu', weights_only=False)
    model = Encoder(config)
    model.load_state_dict(state['model'], strict=True)
    return model, state
