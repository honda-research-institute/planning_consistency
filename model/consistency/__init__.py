from model.consistency.consistency import Consistency
from model.consistency.utils.unet import Unet

def build_consistency(consistency_model_params, mtr_data_cfg):
    unet_model = Unet(
            dim=consistency_model_params['unet_model_channels'],
            dim_mults=consistency_model_params['unet_channel_mult'],
            all_condition_layer_dim=consistency_model_params['all_condition_layer_dim'][-1],
            channels=consistency_model_params['surrounding_k'],
        )

    consistency_model = Consistency(
        unet_model=unet_model,
        sampling_steps=consistency_model_params['sampling_steps'],
        consistency_model_params=consistency_model_params,
        mtr_data_cfg=mtr_data_cfg,
    )

    return consistency_model
