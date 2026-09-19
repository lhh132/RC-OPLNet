"""RC-OPLNet training, with components derived from DGDR/GDRNet."""
import logging
import os
import torch
import modeling.model_manager as models
from modeling.losses import DahLoss
from dataset.data_manager import get_post_FundusAug

ALGORITHMS = ["RC-OPLNet", "RC-OPLNet-Ablation"]


def get_algorithm_class(algorithm_name):
    if algorithm_name not in ALGORITHMS:
        raise NotImplementedError(f"Algorithm not found: {algorithm_name}")
    return RCOPLNet


class RCOPLNet(torch.nn.Module):
    """Complete RC-OPLNet and its explicitly selected component ablations.

    Network, classifier and criterion names preserve existing state dictionaries.
    """
    def __init__(self, num_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.epoch = 0
        if cfg.ALGORITHM == "RC-OPLNet" and not (
            cfg.GDRNET.USE_RC_JDCR and cfg.GDRNET.USE_PROTO_CONTRAST
        ):
            raise ValueError("RC-OPLNet requires RC-JDCR and prototype learning")
        self.network = models.get_net(cfg)
        self.classifier = models.get_classifier(self.network.out_features(), cfg)

        self.optimizer = torch.optim.SGD(
            [{"params":self.network.parameters()},
            {"params":self.classifier.parameters()}],
            lr = cfg.LEARNING_RATE,
            momentum = cfg.MOMENTUM,
            weight_decay = cfg.WEIGHT_DECAY,
            nesterov=True)

        self.fundusAug = get_post_FundusAug(cfg)
        self.criterion = DahLoss(beta= cfg.GDRNET.BETA, max_iteration = cfg.EPOCHS, \
                                training_domains = cfg.DATASET.SOURCE_DOMAINS, temperature = cfg.GDRNET.TEMPERATURE, \
                                scaling_factor = cfg.GDRNET.SCALING_FACTOR,
                                num_classes=cfg.DATASET.NUM_CLASSES,
                                num_domains=len(cfg.DATASET.SOURCE_DOMAINS),
                                domain_class_counts=cfg.DATASET.DOMAIN_CLASS_COUNTS,
                                use_joint_dcr=cfg.GDRNET.USE_JOINT_DCR,
                                joint_dcr_beta=cfg.GDRNET.JOINT_DCR.BETA,
                                joint_dcr_max_weight=cfg.GDRNET.JOINT_DCR.MAX_WEIGHT,
                                use_hcs_dcr=cfg.GDRNET.USE_HCS_DCR,
                                hcs_dcr_beta=cfg.GDRNET.HCS_DCR.BETA,
                                hcs_dcr_tau=cfg.GDRNET.HCS_DCR.TAU,
                                hcs_dcr_min_weight=cfg.GDRNET.HCS_DCR.MIN_WEIGHT,
                                hcs_dcr_max_weight=cfg.GDRNET.HCS_DCR.MAX_WEIGHT,
                                hcs_dcr_lambda_max=cfg.GDRNET.HCS_DCR.LAMBDA_MAX,
                                hcs_dcr_warmup_ratio=cfg.GDRNET.HCS_DCR.WARMUP_RATIO,
                                use_rc_jdcr=cfg.GDRNET.USE_RC_JDCR,
                                rc_jdcr_beta=cfg.GDRNET.RC_JDCR.BETA,
                                rc_jdcr_tau=cfg.GDRNET.RC_JDCR.TAU,
                                rc_jdcr_a_min=cfg.GDRNET.RC_JDCR.A_MIN,
                                rc_jdcr_residual_clip_log=cfg.GDRNET.RC_JDCR.RESIDUAL_CLIP_LOG,
                                rc_jdcr_min_weight=cfg.GDRNET.RC_JDCR.MIN_WEIGHT,
                                rc_jdcr_max_weight=cfg.GDRNET.RC_JDCR.MAX_WEIGHT,
                                rc_jdcr_warmup_start_ratio=cfg.GDRNET.RC_JDCR.WARMUP_START_RATIO,
                                rc_jdcr_full_weight_ratio=cfg.GDRNET.RC_JDCR.FULL_WEIGHT_RATIO,
                                use_ordinal_loss=cfg.GDRNET.USE_ORDINAL_LOSS,
                                ordinal_weight=cfg.GDRNET.ORDINAL.WEIGHT,
                                use_proto_contrast=cfg.GDRNET.USE_PROTO_CONTRAST,
                                feature_dim=self.network.out_features(),
                                proto_weight=cfg.GDRNET.PROTO.WEIGHT,
                                proto_momentum=cfg.GDRNET.PROTO.MOMENTUM,
                                proto_temperature=cfg.GDRNET.PROTO.TEMPERATURE)


    def img_process(self, img_tensor, mask_tensor, fundusAug):
        
        img_tensor_new, mask_tensor_new = fundusAug['post_aug1'](img_tensor.clone(), mask_tensor.clone())
        img_tensor_new = img_tensor_new * mask_tensor_new
        img_tensor_new = fundusAug['post_aug2'](img_tensor_new)
        img_tensor_ori = fundusAug['post_aug2'](img_tensor)

        return img_tensor_new, img_tensor_ori


    def update(self, minibatch):
        
        image, mask, label, domain = minibatch
        
        self.optimizer.zero_grad()
        
        image_new, image_ori = self.img_process(image, mask, self.fundusAug)
        features_ori = self.network(image_ori)
        features_new = self.network(image_new)
        output_new = self.classifier(features_new)

        loss, loss_dict_iter = self.criterion([output_new], [features_ori, features_new], label, domain)
        
        loss.backward()
        self.optimizer.step()

        return loss_dict_iter


    def update_epoch(self, epoch):
        self.epoch = epoch
        return self.criterion.update_alpha(epoch)


    def save_model(self, log_path):
        logging.info("Saving best model...")
        torch.save(self.network.state_dict(), os.path.join(log_path, 'best_model.pth'))
        torch.save(self.classifier.state_dict(), os.path.join(log_path, 'best_classifier.pth'))


    def renew_model(self, log_path):
        net_path = os.path.join(log_path, 'best_model.pth')
        classifier_path = os.path.join(log_path, 'best_classifier.pth')
        self.network.load_state_dict(torch.load(net_path))
        self.classifier.load_state_dict(torch.load(classifier_path))


    def predict(self, x):
        return self.classifier(self.network(x))


