# Motion Transformer (MTR): https://arxiv.org/abs/2209.13508
# Published at NeurIPS 2022
# Written by Shaoshuai Shi 
# All Rights Reserved


from .mtr_encoder import MTREncoder
from .mtr_encoder_predictor import MTREncoderPredictor

__all__ = {
    'MTREncoder': MTREncoder,
    'MTREncoderPredictor': MTREncoderPredictor,
}


def build_context_encoder(config):
    model = __all__[config.NAME](
        config=config
    )

    return model

def build_context_encoder_predictor(in_channels, config):
    model = __all__['MTREncoderPredictor'](
        in_channels=in_channels,
        config=config
    )

    return model