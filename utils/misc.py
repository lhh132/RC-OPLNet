import sys, os, logging, shutil
from torch.utils.tensorboard import SummaryWriter   
import torch, random
import numpy as np
from pathlib import Path
from collections import Counter
from utils.run_record import resume_run_record, start_run_record
ALL_DATASETS=['APTOS','DEEPDR','FGADR','IDRID','MESSIDOR','RLDR']
ESDG_DATASETS = ['APTOS','DEEPDR','FGADR','IDRID','MESSIDOR','RLDR','DDR','EYEPACS']
ALL_METHODS = ["RC-OPLNet", "RC-OPLNet-Ablation"]


def mark_done(log_path):
    Path(log_path, 'done').touch(exist_ok=True)

def count_samples_per_class(targets, num_classes):
    counts = Counter()
    for y in targets:
        counts[int(y)] += 1
    return [counts[i] if counts[i] else np.inf for i in range(num_classes)]

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def init_log(args, cfg, log_path, train_loader_length, dataset_size):
    assert cfg.ALGORITHM in ALL_METHODS
    if not cfg.RANDOM:
        setup_seed(cfg.SEED)
        
    init_output_foler(cfg, log_path)
    writer = SummaryWriter(os.path.join(log_path, 'tensorboard'))
    writer.add_text('config', str(args))
    logging.basicConfig(filename=log_path + '/log.txt', level=logging.INFO,format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info("{} iterations per epoch".format(train_loader_length))
    logging.info("We have {} images in train set, {} images in val set, and {} images in test set.".format(dataset_size[0], dataset_size[1], dataset_size[2]))
    logging.info(str(args))
    logging.info(str(cfg))
    if cfg.RESUME:
        resume_run_record(log_path, args, cfg)
    else:
        start_run_record(log_path, args, cfg, dataset_size)
    return writer

def init_output_foler(cfg, log_path):
    if os.path.isdir(log_path):
        if os.path.exists(os.path.join(log_path, 'done')):
            print('Already trained, exit')
            exit()
        if cfg.RESUME:
            checkpoint = os.path.join(log_path, 'checkpoints', 'last.pth')
            if not os.path.isfile(checkpoint):
                raise FileNotFoundError(
                    'Cannot resume without checkpoints/last.pth: {}'.format(log_path)
                )
        elif cfg.OVERRIDE:
            shutil.rmtree(log_path)
        else:
            shutil.rmtree(log_path)
    else:
        if cfg.RESUME:
            raise FileNotFoundError('Cannot resume missing run: {}'.format(log_path))
        os.makedirs(log_path)

def get_scheduler(optimizer, max_epoch):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch * 0.5], gamma=0.1)
    return scheduler

def update_writer(writer, epoch, scheduler, loss_avg, component_avgs=None):
    logging.info('epoch: {}, total loss: {}'.format(epoch, loss_avg.mean()))
    writer.add_scalar('info/lr', scheduler.get_last_lr()[0], epoch) 
    writer.add_scalar('info/loss', loss_avg.mean(), epoch)
    if component_avgs:
        component_means = {
            name: counter.mean()
            for name, counter in sorted(component_avgs.items())
        }
        logging.info(
            "epoch: %d, loss components: %s",
            epoch,
            ", ".join(
                f"{name}={value:.6f}"
                for name, value in component_means.items()
            ),
        )
        for name, value in component_means.items():
            writer.add_scalar(f"loss/{name}", value, epoch)

    
class LossCounter:
    def __init__(self, start = 0):
        self.sum = start
        self.iteration = 0
    def update(self, num):
        self.sum += num
        self.iteration += 1
    def mean(self):
        return self.sum * 1.0 / self.iteration
