"""
This code is partially borrowed from https://github.com/HobbitLong/SupContrast
"""
from __future__ import print_function
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .gdrnet_components import (
    CrossDomainPrototypeContrast,
    HierarchicalConfidenceShrunkDCR,
    JointDomainClassReweighting,
    OrdinalDistributionLoss,
    ResidualConfidenceCalibratedJDCR,
)

# Our loss function
class DahLoss(nn.Module):
    def __init__(
        self,
        max_iteration,
        training_domains,
        beta=0.8,
        scaling_factor=4,
        alpha=1,
        temperature=0.07,
        num_classes=5,
        num_domains=None,
        domain_class_counts=None,
        use_joint_dcr=False,
        joint_dcr_beta=0.999,
        joint_dcr_max_weight=10.0,
        use_hcs_dcr=False,
        hcs_dcr_beta=0.999,
        hcs_dcr_tau=50.0,
        hcs_dcr_min_weight=0.5,
        hcs_dcr_max_weight=3.0,
        hcs_dcr_lambda_max=0.5,
        hcs_dcr_warmup_ratio=0.2,
        use_rc_jdcr=False,
        rc_jdcr_beta=0.999,
        rc_jdcr_tau=50.0,
        rc_jdcr_a_min=0.5,
        rc_jdcr_residual_clip_log=math.log(4.0),
        rc_jdcr_min_weight=0.25,
        rc_jdcr_max_weight=4.0,
        rc_jdcr_warmup_start_ratio=0.05,
        rc_jdcr_full_weight_ratio=0.20,
        use_ordinal_loss=False,
        ordinal_weight=1.0,
        use_proto_contrast=False,
        feature_dim=None,
        proto_weight=1.0,
        proto_momentum=0.9,
        proto_temperature=0.1,
    ) -> None:
        super(DahLoss, self).__init__()
        self.max_iteration = max_iteration
        self.training_domains = training_domains
        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.temperature = temperature
        self.use_joint_dcr = bool(use_joint_dcr)
        self.use_hcs_dcr = bool(use_hcs_dcr)
        self.use_rc_jdcr = bool(use_rc_jdcr)
        self.use_ordinal_loss = bool(use_ordinal_loss)
        self.use_proto_contrast = bool(use_proto_contrast)
        if self.use_joint_dcr and self.use_hcs_dcr:
            raise ValueError("JDCR and HCS-DCR are mutually exclusive")
        if self.use_rc_jdcr and (
            self.use_joint_dcr
            or self.use_hcs_dcr
        ):
            raise ValueError(
                "RC-JDCR is mutually exclusive with JDCR and HCS-DCR"
            )
        if not 0 <= hcs_dcr_lambda_max <= 1:
            raise ValueError("hcs_dcr_lambda_max must be in [0, 1]")
        if not 0 <= hcs_dcr_warmup_ratio <= 1:
            raise ValueError("hcs_dcr_warmup_ratio must be in [0, 1]")
        self.hcs_dcr_lambda_max = float(hcs_dcr_lambda_max)
        self.hcs_dcr_warmup_ratio = float(hcs_dcr_warmup_ratio)
        self.hcs_lambda = 0.0
        if ordinal_weight < 0:
            raise ValueError("ordinal_weight must be non-negative")
        if proto_weight < 0:
            raise ValueError("proto_weight must be non-negative")
        self.ordinal_weight = float(ordinal_weight)
        self.proto_weight = float(proto_weight)
        
        self.domain_num_dict = {'MESSIDOR': 1744,
                                'IDRID': 516,
                                'DEEPDR': 2000,
                                'FGADR': 1842,
                                'APTOS': 3662,
                                'RLDR': 1593}
        
        self.label_num_dict = {'MESSIDOR': [1016, 269, 347, 75, 35],
                                'IDRID': [175, 26, 163, 89, 60],
                                'DEEPDR': [917, 214, 402, 353, 113],
                                'FGADR': [100, 211, 595, 646, 286],
                                'APTOS': [1804, 369, 999, 192, 294],
                                'RLDR': [165, 336, 929, 98, 62]}

        domain_prob, label_prob = self.get_domain_label_prob()
        domain_prob, label_prob = self.multinomial_soomthing(
            domain_prob, label_prob, self.beta
        )
        self.register_buffer("domain_prob", domain_prob)
        self.register_buffer("label_prob", label_prob)

        self.UnsupLoss = SupConLoss(temperature = self.temperature, reduction='none')
        self.SupLoss = nn.CrossEntropyLoss(reduction='none')
        self.joint_dcr = None
        self.hcs_dcr = None
        self.rc_jdcr = None
        self.ordinal_loss = None
        self.proto_contrast = None

        if self.use_joint_dcr:
            if domain_class_counts is None:
                raise ValueError(
                    "domain_class_counts are required when JDCR is enabled"
                )
            self.joint_dcr = JointDomainClassReweighting(
                domain_class_counts,
                beta=joint_dcr_beta,
                max_weight=joint_dcr_max_weight,
            )
        if self.use_hcs_dcr:
            if domain_class_counts is None:
                raise ValueError(
                    "domain_class_counts are required when HCS-DCR is enabled"
                )
            self.hcs_dcr = HierarchicalConfidenceShrunkDCR(
                domain_class_counts,
                beta=hcs_dcr_beta,
                tau=hcs_dcr_tau,
                min_weight=hcs_dcr_min_weight,
                max_weight=hcs_dcr_max_weight,
            )
            logging.info("HCS-DCR weights: %s", self.hcs_dcr.summary())
        if self.use_rc_jdcr:
            if domain_class_counts is None:
                raise ValueError(
                    "domain_class_counts are required when RC-JDCR is enabled"
                )
            self.rc_jdcr = ResidualConfidenceCalibratedJDCR(
                domain_class_counts,
                beta=rc_jdcr_beta,
                tau=rc_jdcr_tau,
                a_min=rc_jdcr_a_min,
                residual_clip_log=rc_jdcr_residual_clip_log,
                min_weight=rc_jdcr_min_weight,
                max_weight=rc_jdcr_max_weight,
                warmup_start_ratio=rc_jdcr_warmup_start_ratio,
                full_weight_ratio=rc_jdcr_full_weight_ratio,
            )
            logging.info("RC-JDCR weights: %s", self.rc_jdcr.summary())
        if self.use_ordinal_loss:
            self.ordinal_loss = OrdinalDistributionLoss(num_classes)
        if self.use_proto_contrast:
            if feature_dim is None:
                raise ValueError(
                    "feature_dim is required when prototype contrast is enabled"
                )
            if num_domains is None:
                raise ValueError(
                    "num_domains is required when prototype contrast is enabled"
                )
            self.proto_contrast = CrossDomainPrototypeContrast(
                num_classes=num_classes,
                feature_dim=feature_dim,
                num_domains=num_domains,
                momentum=proto_momentum,
                temperature=proto_temperature,
            )

    def get_domain_label_prob(self):
        source_domain_num_list = torch.Tensor([self.domain_num_dict[domain] for domain in self.training_domains])
        source_domain_num = torch.sum(source_domain_num_list)
        domain_prob = source_domain_num_list / source_domain_num

        label_num_list = torch.Tensor([self.label_num_dict[domain] for domain in  self.training_domains]).sum(dim=0)
        label_num = torch.sum(label_num_list)
        label_prob = label_num_list / label_num

        return domain_prob, label_prob

    def multinomial_soomthing(self, domain_prob, label_prob, beta = 0.8):
        domain_prob = torch.pow(domain_prob, beta)
        label_prob = torch.pow(label_prob, beta)

        domain_prob = domain_prob / torch.sum(domain_prob)
        label_prob = label_prob / torch.sum(label_prob)

        return domain_prob, label_prob

    def get_weights(self, labels, domains):
        domain_prob = torch.index_select(self.domain_prob, 0, domains)
        domain_weight = 1 / domain_prob
        class_prob = torch.index_select(self.label_prob, 0, labels)
        class_weight = 1 / class_prob

        return domain_weight, class_weight

    def _legacy_classification_loss(self, output, labels, domains):
        domain_weight, class_weight = self.get_weights(labels, domains)
        loss_sup = sum(self.SupLoss(item, labels) for item in output)
        return torch.mean(
            loss_sup * class_weight * domain_weight
        ) / (torch.mean(domain_weight) * torch.mean(class_weight))
                            
    def forward(self, output, features, labels, domains):
        
        loss_dict = {}

        features_ori, features_new = features

        if self.use_rc_jdcr:
            loss_sup = sum(
                self.rc_jdcr(op_item, labels, domains)
                for op_item in output
            )
            loss_dict['loss_rc_jdcr'] = loss_sup.item()
            loss_dict['rc_jdcr_schedule'] = self.rc_jdcr.schedule
            current = self.rc_jdcr.current_weights
            observed = self.rc_jdcr.counts > 0
            loss_dict['rc_jdcr_weight_min'] = float(
                current[observed].min().item()
            )
            loss_dict['rc_jdcr_weight_max'] = float(
                current[observed].max().item()
            )
        elif self.use_joint_dcr:
            loss_sup = sum(
                self.joint_dcr(op_item, labels, domains)
                for op_item in output
            )
            loss_dict['loss_joint_dcr'] = loss_sup.item()
        else:
            loss_legacy = self._legacy_classification_loss(
                output, labels, domains
            )
            loss_sup = loss_legacy
            if self.use_hcs_dcr:
                loss_hcs = sum(
                    self.hcs_dcr(op_item, labels, domains)
                    for op_item in output
                )
                loss_sup = (
                    (1.0 - self.hcs_lambda) * loss_legacy
                    + self.hcs_lambda * loss_hcs
                )
                loss_dict['loss_legacy'] = loss_legacy.item()
                loss_dict['loss_hcs_dcr'] = loss_hcs.item()
                loss_dict['hcs_lambda'] = self.hcs_lambda

        features_multi = torch.stack([features_ori, features_new], dim = 1)
        features_multi = F.normalize(features_multi, p=2, dim=2)      
        
        loss_unsup = torch.mean(self.UnsupLoss(features_multi))
        classification_loss = loss_sup
        if self.use_ordinal_loss:
            loss_ordinal = sum(
                self.ordinal_loss(op_item, labels) for op_item in output
            )
            classification_loss = (
                classification_loss
                + self.ordinal_weight * loss_ordinal
            )
            loss_dict['loss_ordinal'] = loss_ordinal.item()

        loss = (
            (1 - self.alpha) * classification_loss
            + self.alpha * loss_unsup / self.scaling_factor
        )
        if self.use_proto_contrast:
            proto_features = (features_ori + features_new) / 2
            loss_proto = self.proto_contrast(
                proto_features, labels, domains
            )
            loss = loss + self.proto_weight * loss_proto
            loss_dict['loss_proto'] = loss_proto.item()

        loss_dict['loss'] = loss.item()
        loss_dict['loss_sup'] = loss_sup.item()
        loss_dict['loss_unsup'] = loss_unsup.item()
        
        return loss, loss_dict

    def update_alpha(self, iteration):
        self.alpha = 1 - iteration / self.max_iteration
        if self.use_hcs_dcr:
            warmup_epochs = max(
                1,
                math.ceil(
                    self.max_iteration * self.hcs_dcr_warmup_ratio
                ),
            )
            progress = min(
                1.0, max(0.0, iteration / warmup_epochs)
            )
            self.hcs_lambda = self.hcs_dcr_lambda_max * progress
        if self.use_rc_jdcr:
            schedule = self.rc_jdcr.set_epoch(
                iteration, self.max_iteration
            )
            current = self.rc_jdcr.current_weights
            observed = self.rc_jdcr.counts > 0
            logging.info(
                "RC-JDCR schedule: epoch=%d schedule=%.6f min=%.6f max=%.6f",
                iteration,
                schedule,
                current[observed].min().item(),
                current[observed].max().item(),
            )
        return self.alpha

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, reduction = 'mean'):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
                
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        
        if self.reduction == 'mean':
            loss = loss.view(anchor_count, batch_size).mean()
        else:
            loss = loss.view(anchor_count, batch_size)

        return loss
