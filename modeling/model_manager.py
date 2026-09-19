
from .resnet import resnet18, resnet50, resnet101
import torch

def get_net(cfg):
    if cfg.ALGORITHM not in ("RC-OPLNet", "RC-OPLNet-Ablation"):
        raise ValueError("Unsupported algorithm")
    return get_backbone(cfg)

def get_backbone(cfg):
    if cfg.BACKBONE == 'resnet18':
        model = resnet18(pretrained=True)
    elif cfg.BACKBONE == 'resnet50':
        model = resnet50(pretrained=True)
    elif cfg.BACKBONE == 'resnet101':
        model = resnet101(pretrained=True)
    else:
        raise ValueError('Wrong type')
    return model

def get_classifier(out_feature_size, cfg):
    return torch.nn.Linear(out_feature_size, cfg.DATASET.NUM_CLASSES)
