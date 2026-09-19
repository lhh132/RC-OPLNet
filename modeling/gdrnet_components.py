import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class JointDomainClassReweighting(nn.Module):
    """Effective-number weighting for observed domain-class pairs."""

    def __init__(self, counts, beta=0.999, max_weight=10.0):
        super().__init__()
        if not 0 <= beta < 1:
            raise ValueError("beta must satisfy 0 <= beta < 1")
        if max_weight <= 0:
            raise ValueError("max_weight must be greater than zero")

        counts = torch.as_tensor(counts, dtype=torch.float32)
        if counts.ndim != 2:
            raise ValueError("counts must be a two-dimensional matrix")
        if torch.any(counts < 0):
            raise ValueError("counts must be non-negative")
        if not torch.any(counts > 0):
            raise ValueError("counts must contain an observed pair")

        observed = counts > 0
        weights = torch.zeros_like(counts)
        if beta == 0:
            weights[observed] = 1.0
        else:
            effective_number = 1.0 - torch.pow(beta, counts[observed])
            weights[observed] = (1.0 - beta) / effective_number

        weights[observed] /= weights[observed].mean()
        weights[observed] = weights[observed].clamp(max=max_weight)
        weights[observed] /= weights[observed].mean()
        self.register_buffer("weights", weights)

    def forward(self, logits, labels, domains):
        if logits.ndim != 2:
            raise ValueError("logits must be a two-dimensional tensor")
        if labels.ndim != 1 or domains.ndim != 1:
            raise ValueError("labels and domains must be one-dimensional")
        if logits.shape[0] != labels.shape[0] or labels.shape != domains.shape:
            raise ValueError("batch dimensions must match")
        if logits.shape[1] != self.weights.shape[1]:
            raise ValueError("logit class count does not match count matrix")
        if labels.numel() and (
            labels.min().item() < 0
            or labels.max().item() >= self.weights.shape[1]
        ):
            raise ValueError("class index is outside count matrix")
        if domains.numel() and (
            domains.min().item() < 0
            or domains.max().item() >= self.weights.shape[0]
        ):
            raise ValueError("domain index is outside count matrix")

        sample_weights = self.weights[domains.long(), labels.long()]
        if torch.any(sample_weights <= 0):
            raise ValueError("batch contains an unobserved domain-class pair")
        sample_losses = F.cross_entropy(logits, labels, reduction="none")
        return torch.sum(sample_losses * sample_weights) / torch.sum(
            sample_weights
        )


class HierarchicalConfidenceShrunkDCR(nn.Module):
    """Bounded domain-class weights shrunk toward marginal statistics."""

    def __init__(
        self,
        counts,
        beta=0.999,
        tau=50.0,
        min_weight=0.5,
        max_weight=3.0,
        projection_steps=64,
    ):
        super().__init__()
        if not 0 <= beta < 1:
            raise ValueError("beta must satisfy 0 <= beta < 1")
        if tau <= 0:
            raise ValueError("tau must be greater than zero")
        if min_weight <= 0:
            raise ValueError("min_weight must be greater than zero")
        if not min_weight <= 1 <= max_weight:
            raise ValueError(
                "weight bounds must satisfy min_weight <= 1 <= max_weight"
            )
        if projection_steps <= 0:
            raise ValueError("projection_steps must be greater than zero")

        counts = torch.as_tensor(counts, dtype=torch.float32)
        if counts.ndim != 2:
            raise ValueError("counts must be a two-dimensional matrix")
        if torch.any(counts < 0):
            raise ValueError("counts must be non-negative")
        observed = counts > 0
        if not torch.any(observed):
            raise ValueError("counts must contain an observed pair")

        domain_counts = counts.sum(dim=1)
        class_counts = counts.sum(dim=0)
        domain_weights = self._effective_weights(domain_counts, beta)
        class_weights = self._effective_weights(class_counts, beta)
        joint_weights = self._effective_weights(counts, beta)

        base_weights = torch.sqrt(
            domain_weights[:, None] * class_weights[None, :]
        )
        base_weights = self._normalize_observed(base_weights, observed)
        confidence = counts / (counts + float(tau))

        shrunk = torch.zeros_like(counts)
        shrunk[observed] = torch.exp(
            (1.0 - confidence[observed])
            * torch.log(base_weights[observed])
            + confidence[observed]
            * torch.log(joint_weights[observed])
        )
        weights = self._bounded_mean_projection(
            shrunk,
            observed,
            float(min_weight),
            float(max_weight),
            int(projection_steps),
        )

        self.register_buffer("counts", counts)
        self.register_buffer("domain_weights", domain_weights)
        self.register_buffer("class_weights", class_weights)
        self.register_buffer("joint_weights", joint_weights)
        self.register_buffer("base_weights", base_weights)
        self.register_buffer("confidence", confidence)
        self.register_buffer("shrunk_weights", shrunk)
        self.register_buffer("weights", weights)

    @staticmethod
    def _normalize_observed(values, observed):
        result = torch.zeros_like(values)
        result[observed] = values[observed] / values[observed].mean()
        return result

    @classmethod
    def _effective_weights(cls, counts, beta):
        observed = counts > 0
        weights = torch.zeros_like(counts)
        if beta == 0:
            weights[observed] = 1.0
        else:
            effective_number = 1.0 - torch.pow(beta, counts[observed])
            weights[observed] = (1.0 - beta) / effective_number
        return cls._normalize_observed(weights, observed)

    @staticmethod
    def _bounded_mean_projection(
        values, observed, min_weight, max_weight, steps
    ):
        positive = values[observed]
        low = positive.new_tensor(0.0)
        high = positive.new_tensor(max_weight) / positive.min()
        for _ in range(steps):
            middle = (low + high) / 2.0
            candidate_mean = torch.clamp(
                middle * positive,
                min=min_weight,
                max=max_weight,
            ).mean()
            if candidate_mean < 1.0:
                low = middle
            else:
                high = middle
        result = torch.zeros_like(values)
        scale = (low + high) / 2.0
        result[observed] = torch.clamp(
            scale * positive,
            min=min_weight,
            max=max_weight,
        )
        return result

    def summary(self):
        observed = self.counts > 0
        values = self.weights[observed]
        return {
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "mean": float(values.mean().item()),
            "counts": self.counts.detach().cpu().tolist(),
            "weights": self.weights.detach().cpu().tolist(),
        }

    def forward(self, logits, labels, domains):
        if logits.ndim != 2:
            raise ValueError("logits must be a two-dimensional tensor")
        if labels.ndim != 1 or domains.ndim != 1:
            raise ValueError("labels and domains must be one-dimensional")
        if logits.shape[0] != labels.shape[0] or labels.shape != domains.shape:
            raise ValueError("batch dimensions must match")
        if logits.shape[1] != self.weights.shape[1]:
            raise ValueError("logit class count does not match count matrix")
        if labels.numel() and (
            labels.min().item() < 0
            or labels.max().item() >= self.weights.shape[1]
        ):
            raise ValueError("class index is outside count matrix")
        if domains.numel() and (
            domains.min().item() < 0
            or domains.max().item() >= self.weights.shape[0]
        ):
            raise ValueError("domain index is outside count matrix")
        sample_weights = self.weights[domains.long(), labels.long()]
        if torch.any(sample_weights <= 0):
            raise ValueError("batch contains an unobserved domain-class pair")
        losses = F.cross_entropy(logits, labels, reduction="none")
        return torch.sum(losses * sample_weights) / torch.sum(sample_weights)


class ResidualConfidenceCalibratedJDCR(nn.Module):
    """Confidence-calibrate only the joint/marginal log residual."""

    def __init__(
        self,
        counts,
        beta=0.999,
        tau=50.0,
        a_min=0.5,
        residual_clip_log=math.log(4.0),
        min_weight=0.25,
        max_weight=4.0,
        warmup_start_ratio=0.05,
        full_weight_ratio=0.20,
        projection_steps=64,
    ):
        super().__init__()
        if not 0 <= beta < 1:
            raise ValueError("beta must satisfy 0 <= beta < 1")
        if tau <= 0:
            raise ValueError("tau must be greater than zero")
        if not 0 <= a_min <= 1:
            raise ValueError("a_min must be in [0, 1]")
        if residual_clip_log < 0:
            raise ValueError("residual_clip_log must be non-negative")
        if not 0 < min_weight <= 1 <= max_weight:
            raise ValueError(
                "weight bounds must satisfy 0 < min_weight <= 1 <= max_weight"
            )
        if not 0 <= warmup_start_ratio < full_weight_ratio <= 1:
            raise ValueError(
                "warmup ratios must satisfy 0 <= start < full <= 1"
            )
        if projection_steps <= 0:
            raise ValueError("projection_steps must be greater than zero")

        counts = torch.as_tensor(counts, dtype=torch.float32)
        if counts.ndim != 2:
            raise ValueError("counts must be a two-dimensional matrix")
        if torch.any(counts < 0):
            raise ValueError("counts must be non-negative")
        observed = counts > 0
        if not torch.any(observed):
            raise ValueError("counts must contain an observed pair")
        class_counts = counts.sum(dim=0)
        if torch.any(class_counts <= 0):
            raise ValueError("every declared class count must be positive")
        domain_counts = counts.sum(dim=1)
        if torch.any(domain_counts <= 0):
            raise ValueError("every declared domain count must be positive")

        marginal = self._effective_weights(class_counts, beta)
        marginal = self._frequency_normalize(marginal, class_counts)
        joint = self._effective_weights(counts, beta)
        joint = self._frequency_normalize(joint, counts)
        expanded_marginal = marginal.unsqueeze(0).expand_as(counts)

        residual = torch.zeros_like(counts)
        residual[observed] = (
            torch.log(joint[observed])
            - torch.log(expanded_marginal[observed])
        )
        confidence = torch.zeros_like(counts)
        confidence[observed] = float(a_min) + (1.0 - float(a_min)) * (
            counts[observed] / (counts[observed] + float(tau))
        )
        candidate = torch.zeros_like(counts)
        candidate[observed] = torch.exp(
            torch.log(expanded_marginal[observed])
            + confidence[observed]
            * residual[observed].clamp(
                min=-float(residual_clip_log),
                max=float(residual_clip_log),
            )
        )
        weights = self._bounded_frequency_projection(
            candidate,
            counts,
            float(min_weight),
            float(max_weight),
            int(projection_steps),
        )

        self.warmup_start_ratio = float(warmup_start_ratio)
        self.full_weight_ratio = float(full_weight_ratio)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.schedule = 0.0
        self.register_buffer("counts", counts)
        self.register_buffer("marginal_weights", marginal)
        self.register_buffer("joint_weights", joint)
        self.register_buffer("residual", residual)
        self.register_buffer("confidence", confidence)
        self.register_buffer("candidate_weights", candidate)
        self.register_buffer("weights", weights)

    @staticmethod
    def _effective_weights(counts, beta):
        observed = counts > 0
        result = torch.zeros_like(counts)
        if beta == 0:
            result[observed] = 1.0
        else:
            result[observed] = (1.0 - beta) / (
                1.0 - torch.pow(beta, counts[observed])
            )
        return result

    @staticmethod
    def _frequency_normalize(values, counts):
        total = counts.sum()
        mean = (values * counts).sum() / total
        return values / mean

    @staticmethod
    def _bounded_frequency_projection(
        values, counts, min_weight, max_weight, steps
    ):
        observed = counts > 0
        positive = values[observed]
        frequencies = counts[observed]
        low = positive.new_tensor(0.0)
        high = positive.new_tensor(max_weight) / positive.min()
        for _ in range(steps):
            middle = (low + high) / 2.0
            projected = torch.clamp(
                middle * positive,
                min=min_weight,
                max=max_weight,
            )
            weighted_mean = (
                projected * frequencies
            ).sum() / frequencies.sum()
            if weighted_mean < 1.0:
                low = middle
            else:
                high = middle
        result = torch.zeros_like(values)
        scale = (low + high) / 2.0
        result[observed] = torch.clamp(
            scale * positive,
            min=min_weight,
            max=max_weight,
        )
        return result

    @property
    def current_weights(self):
        observed = self.counts > 0
        current = torch.zeros_like(self.weights)
        current[observed] = torch.exp(
            self.schedule * torch.log(self.weights[observed])
        )
        return current

    def set_progress(self, progress):
        progress = min(1.0, max(0.0, float(progress)))
        if progress <= self.warmup_start_ratio:
            self.schedule = 0.0
        elif progress >= self.full_weight_ratio:
            self.schedule = 1.0
        else:
            self.schedule = (
                (progress - self.warmup_start_ratio)
                / (self.full_weight_ratio - self.warmup_start_ratio)
            )
        return self.schedule

    def set_epoch(self, epoch_index, total_epochs):
        total_epochs = max(int(total_epochs), 1)
        last_epoch_index = max(total_epochs - 1, 0)
        epoch_index = min(max(int(epoch_index), 0), last_epoch_index)
        denominator = max(last_epoch_index, 1)
        return self.set_progress(float(epoch_index) / denominator)

    def _validate_batch(self, logits, labels, domains):
        if logits.ndim != 2:
            raise ValueError("logits must be a two-dimensional tensor")
        if labels.ndim != 1 or domains.ndim != 1:
            raise ValueError("labels and domains must be one-dimensional")
        if logits.shape[0] != labels.shape[0] or labels.shape != domains.shape:
            raise ValueError("batch dimensions must match")
        if logits.shape[1] != self.weights.shape[1]:
            raise ValueError("logit class count does not match count matrix")
        if labels.numel() and (
            labels.min().item() < 0
            or labels.max().item() >= self.weights.shape[1]
        ):
            raise ValueError("class index is outside count matrix")
        if domains.numel() and (
            domains.min().item() < 0
            or domains.max().item() >= self.weights.shape[0]
        ):
            raise ValueError("domain index is outside count matrix")

    def summary(self):
        observed = self.counts > 0
        values = self.weights[observed]
        frequencies = self.counts[observed]
        weighted_mean = (values * frequencies).sum() / frequencies.sum()
        quantile_levels = values.new_tensor([0.10, 0.25, 0.50, 0.75, 0.90])
        quantiles = torch.quantile(values, quantile_levels)
        lower = torch.isclose(
            values, values.new_tensor(self.min_weight), atol=1e-6
        )
        upper = torch.isclose(
            values, values.new_tensor(self.max_weight), atol=1e-6
        )
        per_domain = (
            (self.weights * self.counts).sum(dim=1)
            / self.counts.sum(dim=1)
        )
        per_class = (
            (self.weights * self.counts).sum(dim=0)
            / self.counts.sum(dim=0)
        )
        joint_difference = torch.abs(
            self.weights[observed] - self.joint_weights[observed]
        )
        relative_joint_difference = (
            joint_difference / self.joint_weights[observed]
        )
        weighted_confidence = (
            self.confidence[observed] * frequencies
        ).sum() / frequencies.sum()
        return {
            "min": float(values.min().item()),
            "max": float(values.max().item()),
            "frequency_weighted_mean": float(weighted_mean.item()),
            "quantiles": {
                str(float(level.item())): float(value.item())
                for level, value in zip(quantile_levels, quantiles)
            },
            "lower_saturation_rate": float(
                (frequencies[lower].sum() / frequencies.sum()).item()
            ),
            "upper_saturation_rate": float(
                (frequencies[upper].sum() / frequencies.sum()).item()
            ),
            "per_domain_mean_weight": per_domain.detach().cpu().tolist(),
            "per_class_mean_weight": per_class.detach().cpu().tolist(),
            "frequency_weighted_confidence": float(
                weighted_confidence.item()
            ),
            "mean_absolute_joint_difference": float(
                joint_difference.mean().item()
            ),
            "max_absolute_joint_difference": float(
                joint_difference.max().item()
            ),
            "mean_relative_joint_difference": float(
                relative_joint_difference.mean().item()
            ),
            "max_relative_joint_difference": float(
                relative_joint_difference.max().item()
            ),
            "counts": self.counts.detach().cpu().tolist(),
            "marginal_weights": self.marginal_weights.detach().cpu().tolist(),
            "joint_weights": self.joint_weights.detach().cpu().tolist(),
            "residual": self.residual.detach().cpu().tolist(),
            "confidence": self.confidence.detach().cpu().tolist(),
            "weights": self.weights.detach().cpu().tolist(),
        }

    def forward(self, logits, labels, domains):
        self._validate_batch(logits, labels, domains)
        sample_weights = self.current_weights[
            domains.long(), labels.long()
        ]
        if torch.any(sample_weights <= 0):
            raise ValueError("batch contains an unobserved domain-class pair")
        losses = F.cross_entropy(logits, labels, reduction="none")
        return torch.sum(losses * sample_weights) / torch.sum(sample_weights)


class OrdinalDistributionLoss(nn.Module):
    """Squared CDF distance for ordered class distributions."""

    def __init__(self, num_classes):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = int(num_classes)

    def forward(self, logits, labels):
        if logits.ndim != 2:
            raise ValueError("logits must be a two-dimensional tensor")
        if logits.shape[1] != self.num_classes:
            raise ValueError("logit class count does not match num_classes")
        if labels.ndim != 1 or logits.shape[0] != labels.shape[0]:
            raise ValueError("batch dimensions must match")
        if labels.numel() and (
            labels.min().item() < 0
            or labels.max().item() >= self.num_classes
        ):
            raise ValueError("label index is outside class range")

        probabilities = torch.softmax(logits, dim=1)
        targets = F.one_hot(
            labels.long(), num_classes=self.num_classes
        ).to(probabilities.dtype)
        prediction_cdf = probabilities.cumsum(dim=1)
        target_cdf = targets.cumsum(dim=1)
        return (prediction_cdf - target_cdf).pow(2).mean()


class CrossDomainPrototypeContrast(nn.Module):
    """EMA class prototypes shared across source domains."""

    def __init__(
        self,
        num_classes,
        feature_dim,
        num_domains,
        momentum=0.9,
        temperature=0.1,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be greater than zero")
        if num_domains < 2:
            raise ValueError("num_domains must be at least 2")
        if not 0 <= momentum < 1:
            raise ValueError("momentum must satisfy 0 <= momentum < 1")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero")

        self.num_classes = int(num_classes)
        self.feature_dim = int(feature_dim)
        self.num_domains = int(num_domains)
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.register_buffer(
            "prototypes", torch.zeros(num_classes, feature_dim)
        )
        self.register_buffer(
            "initialized", torch.zeros(num_classes, dtype=torch.bool)
        )
        self.register_buffer(
            "domains_seen",
            torch.zeros(num_classes, num_domains, dtype=torch.bool),
        )

    @property
    def class_is_cross_domain(self):
        return self.domains_seen.sum(dim=1) >= 2

    def _validate_forward_inputs(self, features, labels, domains):
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                "features must be a two-dimensional tensor with feature_dim"
            )
        if labels.ndim != 1 or domains.ndim != 1:
            raise ValueError("labels and domains must be one-dimensional")
        if features.shape[0] != labels.shape[0] or labels.shape != domains.shape:
            raise ValueError("batch dimensions must match")
        if labels.numel() and (
            labels.min().item() < 0
            or labels.max().item() >= self.num_classes
        ):
            raise ValueError("class index is outside prototype range")
        if domains.numel() and (
            domains.min().item() < 0
            or domains.max().item() >= self.num_domains
        ):
            raise ValueError("domain index is outside prototype range")

    def ordered_margin_loss(self, features, labels):
        """Penalize overly similar current class centers by grade distance."""
        if features.ndim != 2 or features.shape[0] != labels.shape[0]:
            raise ValueError("features and labels must have matching batches")
        unique_classes = torch.unique(labels, sorted=True)
        if unique_classes.numel() < 2:
            return features.sum() * 0.0

        centers = torch.stack(
            [
                F.normalize(
                    features[labels == class_id].mean(dim=0),
                    dim=0,
                )
                for class_id in unique_classes
            ]
        )
        penalties = []
        for left in range(unique_classes.numel()):
            for right in range(left + 1, unique_classes.numel()):
                grade_distance = (
                    unique_classes[right] - unique_classes[left]
                ).abs().to(features.dtype)
                required_margin = torch.clamp(
                    grade_distance / (self.num_classes - 1), max=1.0
                )
                similarity = F.cosine_similarity(
                    centers[left], centers[right], dim=0
                )
                penalties.append(
                    F.relu(similarity - (1.0 - required_margin))
                )
        return torch.stack(penalties).mean()

    def _prototype_classification_loss(
        self, features, labels, cross_domain_before
    ):
        valid_samples = (
            cross_domain_before[labels] & self.initialized[labels]
        )
        if not torch.any(valid_samples):
            return features.sum() * 0.0

        prototype_indices = torch.nonzero(
            self.initialized, as_tuple=False
        ).flatten()
        prototypes = F.normalize(
            self.prototypes[prototype_indices].detach(), dim=1
        )
        logits = (
            features[valid_samples] @ prototypes.t()
        ) / self.temperature
        local_targets = torch.full(
            (self.num_classes,),
            -1,
            device=labels.device,
            dtype=torch.long,
        )
        local_targets[prototype_indices] = torch.arange(
            prototype_indices.numel(), device=labels.device
        )
        return F.cross_entropy(
            logits, local_targets[labels[valid_samples]]
        )

    @torch.no_grad()
    def _update_state(self, features, labels, domains):
        for class_id in torch.unique(labels):
            class_index = int(class_id.item())
            class_mask = labels == class_id
            for domain_id in torch.unique(domains[class_mask]):
                self.domains_seen[class_index, int(domain_id.item())] = True

            batch_mean = F.normalize(
                features[class_mask].mean(dim=0), dim=0
            )
            if self.initialized[class_index]:
                updated = (
                    self.momentum * self.prototypes[class_index]
                    + (1.0 - self.momentum) * batch_mean
                )
                self.prototypes[class_index] = F.normalize(updated, dim=0)
            else:
                self.prototypes[class_index] = batch_mean
                self.initialized[class_index] = True

    def forward(self, features, labels, domains):
        self._validate_forward_inputs(features, labels, domains)
        normalized = F.normalize(features, dim=1)
        cross_domain_before = self.class_is_cross_domain.clone()

        classification = self._prototype_classification_loss(
            normalized, labels.long(), cross_domain_before
        )
        eligible = cross_domain_before[labels.long()]
        if torch.any(eligible):
            ordered = self.ordered_margin_loss(
                normalized[eligible], labels.long()[eligible]
            )
        else:
            ordered = normalized.sum() * 0.0

        self._update_state(
            normalized.detach(), labels.long(), domains.long()
        )
        return classification + ordered
