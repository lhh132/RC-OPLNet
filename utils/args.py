import argparse
from pathlib import Path

from configs.defaults import _C as cfg_default
from utils.method_names import validate_gdrnet_switches


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "datasets"


def dataset_config_path(dg_mode):
    names = {
        "DG": "GDRBench.yaml",
        "ESDG": "GDRBench_ESDG.yaml",
    }
    try:
        filename = names[dg_mode]
    except KeyError as error:
        raise ValueError("Wrong type") from error
    path = CONFIG_DIR / filename
    if not path.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {path}")
    return path


def get_args():
    parser = argparse.ArgumentParser(description="RC-OPLNet training and evaluation")
    #parser.add_argument("--root", type=str, default="../DGDATA/", help="path to dataset")
    parser.add_argument("--root", type=str, default="../data/FundusDG", help="path to dataset")
    
    parser.add_argument("--algorithm", type=str, default='RC-OPLNet', choices=('RC-OPLNet', 'RC-OPLNet-Ablation'))
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for RC-OPLNet")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for RC-OPLNet")
    parser.add_argument("--dg_mode", type=str, default='DG', help="DG or ESDG")
    parser.add_argument(
        "--split-profile",
        choices=("official", "strict"),
        default="official",
        help="official image-level split or strict patient-level split",
    )
    
    parser.add_argument("--num_classes", type=int, default=5)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ep", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=None, help="override YAML epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="override YAML batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--max-train-batches", type=int, default=None, help="0 means all")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="0 means all")
    parser.add_argument("--output", type=str, default='test')
    parser.add_argument("--override", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gdrnet-joint-dcr", action="store_true")
    parser.add_argument("--gdrnet-hcs-dcr", action="store_true")
    parser.add_argument("--gdrnet-rc-jdcr", action="store_true")
    parser.add_argument("--rc-jdcr-a-min", type=float, default=None)
    parser.add_argument(
        "--rc-jdcr-residual-clip-log", type=float, default=None
    )
    parser.add_argument("--rc-jdcr-max-weight", type=float, default=None)
    parser.add_argument("--gdrnet-ordinal-loss", action="store_true")
    parser.add_argument("--gdrnet-proto-contrast", action="store_true")
    parser.add_argument("--eval-only", action="store_true", help="evaluate a saved RC-OPLNet checkpoint without training")
    parser.add_argument("--checkpoint-dir", type=str, help="directory containing best_model.pth and best_classifier.pth")
    args = parser.parse_args()
    if args.eval_only and not args.checkpoint_dir:
        parser.error("--eval-only requires --checkpoint-dir")
    if args.eval_only and args.resume:
        parser.error("--eval-only cannot be combined with --resume")
    return args

def setup_cfg(args):
    cfg = cfg_default.clone()
    cfg.RANDOM = args.random
    cfg.SEED = args.seed
    cfg.OUTPUT_PATH = args.output
    cfg.OVERRIDE = args.override
    cfg.DG_MODE = args.dg_mode
    
    cfg.ALGORITHM = args.algorithm
    cfg.BACKBONE = args.backbone
    
    cfg.DATASET.ROOT = args.root
    cfg.DATASET.SOURCE_DOMAINS = args.source_domains
    cfg.DATASET.TARGET_DOMAINS = args.target_domains
    cfg.DATASET.NUM_CLASSES = args.num_classes
    cfg.DATASET.SPLIT_PROFILE = args.split_profile

    cfg.VAL_EPOCH = args.val_ep
    
    cfg.merge_from_file(str(dataset_config_path(args.dg_mode)))

    if cfg.ALGORITHM == "RC-OPLNet":
        cfg.GDRNET.USE_RC_JDCR = True
        cfg.GDRNET.USE_PROTO_CONTRAST = True
        cfg.GDRNET.RC_JDCR.MAX_WEIGHT = 6.0

    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.num_workers is not None:
        cfg.NUM_WORKERS = args.num_workers
    if getattr(args, "max_train_batches", None) is not None:
        cfg.MAX_TRAIN_BATCHES = args.max_train_batches
    if getattr(args, "max_eval_batches", None) is not None:
        cfg.MAX_EVAL_BATCHES = args.max_eval_batches

    cfg.RESUME = bool(getattr(args, "resume", False))

    if getattr(args, "gdrnet_joint_dcr", False):
        cfg.GDRNET.USE_JOINT_DCR = True
    if getattr(args, "gdrnet_hcs_dcr", False):
        cfg.GDRNET.USE_HCS_DCR = True
    if getattr(args, "gdrnet_rc_jdcr", False):
        cfg.GDRNET.USE_RC_JDCR = True
    if getattr(args, "rc_jdcr_a_min", None) is not None:
        cfg.GDRNET.RC_JDCR.A_MIN = args.rc_jdcr_a_min
    if getattr(args, "rc_jdcr_residual_clip_log", None) is not None:
        cfg.GDRNET.RC_JDCR.RESIDUAL_CLIP_LOG = (
            args.rc_jdcr_residual_clip_log
        )
    if getattr(args, "rc_jdcr_max_weight", None) is not None:
        cfg.GDRNET.RC_JDCR.MAX_WEIGHT = args.rc_jdcr_max_weight
    if cfg.GDRNET.RC_JDCR.MAX_WEIGHT < 1.0:
        raise ValueError("RC-JDCR max weight must be at least 1")
    if getattr(args, "gdrnet_ordinal_loss", False):
        cfg.GDRNET.USE_ORDINAL_LOSS = True
    if getattr(args, "gdrnet_proto_contrast", False):
        cfg.GDRNET.USE_PROTO_CONTRAST = True

    validate_gdrnet_switches(cfg)
    validate_domain_config(cfg)
    return cfg


def validate_domain_config(cfg):
    if cfg.ALGORITHM not in {"RC-OPLNet", "RC-OPLNet-Ablation"}:
        raise ValueError("Unsupported algorithm")
    sources = tuple(cfg.DATASET.SOURCE_DOMAINS or ())
    targets = tuple(cfg.DATASET.TARGET_DOMAINS or ())
    if cfg.GDRNET.USE_PROTO_CONTRAST and len(set(sources)) < 2:
        raise ValueError("Prototype learning requires at least two source domains")
    if not sources or not targets:
        raise ValueError("Source and target domains are required")
    if set(sources) & set(targets):
        raise ValueError("target-domain leakage: source and target domains overlap")
