import algorithms
import os
from pathlib import Path
from utils.validate import *
from utils.args import *
from utils.misc import *
from dataset.data_manager import get_dataset
from tqdm import tqdm
from utils.checkpointing import should_save_checkpoint
from utils.result_paths import result_path
from utils.run_record import finish_run_record, write_failure_context
from utils.training_state import load_last_checkpoint, save_last_checkpoint


def evaluate_for_checkpoint(algorithm, val_loader, writer, epoch):
    """Evaluate only source validation data during model selection."""
    return algorithm_validate(algorithm, val_loader, writer, epoch, "val")


def evaluate_final_target(algorithm, test_loader, writer, cfg):
    """Evaluate the held-out target once after loading the best checkpoint."""
    final_epoch = cfg.EPOCHS + cfg.VAL_EPOCH
    return algorithm_validate(
        algorithm, test_loader, writer, final_epoch, "test"
    )


def run():
    args = get_args()
    cfg = setup_cfg(args)
    log_path = os.fspath(result_path(cfg))
    if getattr(args, "eval_only", False):
        checkpoint_dir = Path(args.checkpoint_dir).resolve()
        if Path(log_path).resolve() == checkpoint_dir:
            raise ValueError("Evaluation --output must differ from the checkpoint directory")
        for name in ("best_model.pth", "best_classifier.pth"):
            if not (checkpoint_dir / name).is_file():
                raise FileNotFoundError(checkpoint_dir / name)
    train_loader, val_loader, test_loader, dataset_size = get_dataset(cfg)
    writer = init_log(args, cfg, log_path, len(train_loader), dataset_size)
    failure_context = None
    try:
        algorithm_class = algorithms.get_algorithm_class(cfg.ALGORITHM)
        algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)
        algorithm.cuda()

        if getattr(args, "eval_only", False):
            algorithm.renew_model(str(checkpoint_dir))
            algorithm_validate(algorithm, val_loader, writer, 0, "val")
            algorithm_validate(algorithm, test_loader, writer, 0, "test")
            finish_run_record(log_path, "completed")
            return

        scheduler = get_scheduler(algorithm.optimizer, cfg.EPOCHS)
        start_epoch = 0
        best_performance = float("-inf")
        last_checkpoint = Path(log_path) / "checkpoints" / "last.pth"
        if cfg.RESUME:
            start_epoch, best_performance = load_last_checkpoint(
                last_checkpoint, algorithm, scheduler, cfg
            )
        iterator = tqdm(range(start_epoch, cfg.EPOCHS))
        for i in iterator:
            epoch = i + 1
            loss_avg = LossCounter()
            component_avgs = {}
            for batch_index, (
                image,
                mask,
                label,
                domain,
                img_index,
            ) in enumerate(train_loader):
                algorithm.train()
                minibatch = [
                    image.cuda(),
                    mask.cuda(),
                    label.cuda().long(),
                    domain.cuda().long(),
                ]
                failure_context = (epoch, batch_index, minibatch[2], minibatch[3])
                loss_dict_iter = algorithm.update(minibatch)
                failure_context = None
                loss_avg.update(loss_dict_iter["loss"])
                for name, value in loss_dict_iter.items():
                    if name == "loss" or not isinstance(value, (int, float)):
                        continue
                    component_avgs.setdefault(name, LossCounter()).update(value)
                if (
                    cfg.MAX_TRAIN_BATCHES
                    and batch_index + 1 >= cfg.MAX_TRAIN_BATCHES
                ):
                    break

            algorithm.update_epoch(epoch)
            update_writer(writer, epoch, scheduler, loss_avg, component_avgs)
            scheduler.step()

            if epoch % cfg.VAL_EPOCH == 0:
                val_auc, _ = evaluate_for_checkpoint(
                    algorithm, val_loader, writer, epoch
                )
                if should_save_checkpoint(
                    val_auc, best_performance, epoch, cfg.EPOCHS
                ):
                    best_performance = val_auc
                    algorithm.save_model(log_path)
            save_last_checkpoint(
                last_checkpoint,
                algorithm,
                scheduler,
                epoch,
                best_performance,
                cfg,
            )

        algorithm.renew_model(log_path)
        evaluate_final_target(algorithm, test_loader, writer, cfg)
        finish_run_record(log_path, "completed")
    except BaseException as error:
        if isinstance(error, FloatingPointError) and failure_context is not None:
            write_failure_context(log_path, *failure_context)
        metadata_path = os.path.join(log_path, "run_metadata.json")
        if os.path.isfile(metadata_path):
            finish_run_record(log_path, "failed", error=error)
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    run()
