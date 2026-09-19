COMPONENTS = (
    ("USE_JOINT_DCR", "JDCR"),
    ("USE_HCS_DCR", "HCSDCR"),
    ("USE_RC_JDCR", "RCJDCR"),
    ("USE_ORDINAL_LOSS", "ORD"),
    ("USE_PROTO_CONTRAST", "PROTO"),
)


def enabled_gdrnet_components(cfg):
    return [
        suffix
        for field, suffix in COMPONENTS
        if bool(getattr(cfg.GDRNET, field))
    ]


def validate_gdrnet_switches(cfg):
    enabled = enabled_gdrnet_components(cfg)
    if cfg.GDRNET.USE_JOINT_DCR and cfg.GDRNET.USE_HCS_DCR:
        raise ValueError("JDCR and HCS-DCR switches are mutually exclusive")
    if cfg.GDRNET.USE_RC_JDCR and (
        cfg.GDRNET.USE_JOINT_DCR
        or cfg.GDRNET.USE_HCS_DCR
    ):
        raise ValueError(
            "RC-JDCR is mutually exclusive with JDCR and HCS-DCR"
        )
    if enabled and cfg.ALGORITHM not in {"RC-OPLNet-Ablation", "RC-OPLNet"}:
        raise ValueError(
            "GDRNet improvement switches are only valid with RC-OPLNet or RC-OPLNet-Ablation"
        )
    if cfg.ALGORITHM == "RC-OPLNet" and (
        not cfg.GDRNET.USE_RC_JDCR or not cfg.GDRNET.USE_PROTO_CONTRAST
        or cfg.GDRNET.USE_ORDINAL_LOSS
    ):
        raise ValueError("RC-OPLNet requires RC-JDCR + PROTO; use RC-OPLNet-Ablation for ablations")
    return enabled


def _weight_suffix(value):
    text = format(float(value), "g").replace(".", "p")
    return f"WMAX{text}"


def method_display_name(cfg):
    enabled = validate_gdrnet_switches(cfg)
    if cfg.ALGORITHM == "RC-OPLNet":
        weight = float(cfg.GDRNET.RC_JDCR.MAX_WEIGHT)
        return "RC-OPLNet" if weight == 6.0 else "RC-OPLNet-" + _weight_suffix(weight)
    if cfg.ALGORITHM != "RC-OPLNet-Ablation" or not enabled:
        return cfg.ALGORITHM
    if (
        cfg.GDRNET.USE_RC_JDCR
        and float(cfg.GDRNET.RC_JDCR.MAX_WEIGHT) != 4.0
    ):
        enabled = [
            *enabled,
            _weight_suffix(cfg.GDRNET.RC_JDCR.MAX_WEIGHT),
        ]
    return "-".join(["RC-OPLNet-Ablation", *enabled])
