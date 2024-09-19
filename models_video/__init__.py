from .autoencoder_kl_cond_video import AutoencoderKLVideo
from .unet_video import UNetVideoModel
from .propagation_module import Propagation
from torch.optim.lr_scheduler import LambdaLR

def customized_lr_scheduler(optimizer, warmup_steps=5000): # 5000 from u-vit
    from torch.optim.lr_scheduler import LambdaLR
    def fn(step):
        if warmup_steps > 0:
            return min(step / warmup_steps, 1)
        else:
            return 1
    return LambdaLR(optimizer, fn)


def get_lr_scheduler(optimizer, name, **kwargs):
    if name == 'warmup':
        return customized_lr_scheduler(optimizer, **kwargs)
    elif name == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, **kwargs)
    else:
        raise NotImplementedError(name)
    
def get_models(config_path='./configs/unet_video_config.json'):
    config_path = config_path
    pretrained_model_path = "./pretrained_models/unet_diffusion_pytorch_model.bin"
    return UNetVideoModel.from_pretrained_2d(config_path, pretrained_model_path)

    