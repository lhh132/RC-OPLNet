from pathlib import Path

from utils.method_names import method_display_name

import torch
import logging

from utils.benchmark_metrics import compute_domain_metrics, write_metrics

# validate the algorithm by AUC, accuracy and f1 score on val/test datasets


def benchmark_metadata(cfg, algorithm_name, split, epoch):
    profile = cfg.DATASET.SPLIT_PROFILE
    return {
        'algorithm': algorithm_name,
        'split': split,
        'epoch': int(epoch),
        'split_profile': profile,
        'benchmark': (
            'GDRBench-StrictPatient-v1'
            if profile == 'strict'
            else 'GDRBench-official-image-split'
        ),
        'source_domains': list(cfg.DATASET.SOURCE_DOMAINS),
        'target_domains': list(cfg.DATASET.TARGET_DOMAINS),
    }

def algorithm_validate(algorithm, data_loader, writer, epoch, val_type):
    algorithm.eval()
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        softmax = torch.nn.Softmax(dim=1)
        loss = 0
        label_list = []
        output_list = []
        pred_list = []
        domain_list = []

        for batch_index, (image, label, domain, _) in enumerate(data_loader):
            image = image.cuda()
            label = label.cuda().long()

            output = algorithm.predict(image)
            loss += criterion(output, label).item()

            _, pred = torch.max(output, 1)
            output_sf = softmax(output)

            label_list.append(label.cpu().data.numpy())
            pred_list.append(pred.cpu().data.numpy())
            output_list.append(output_sf.cpu().data.numpy())
            domain_list.append(domain.cpu().data.numpy())
            if algorithm.cfg.MAX_EVAL_BATCHES and batch_index + 1 >= algorithm.cfg.MAX_EVAL_BATCHES:
                break
        
        label = [item for sublist in label_list for item in sublist]
        pred = [item for sublist in pred_list for item in sublist]
        output = [item for sublist in output_list for item in sublist]
        domains = [item for sublist in domain_list for item in sublist]

        metrics = compute_domain_metrics(
            label,
            output,
            domains,
            data_loader.dataset.domain_names,
            num_classes=len(output[0]),
        )
        pooled_metrics = metrics['pooled']
        acc = pooled_metrics['accuracy']
        f1 = pooled_metrics['macro_f1']
        auc_ovo = pooled_metrics['auc_ovo']

        loss = loss / (batch_index + 1)

        if val_type in ['val', 'test']:
            writer.add_scalar('info/{}_accuracy'.format(val_type), acc, epoch)
            writer.add_scalar('info/{}_loss'.format(val_type), loss, epoch)
            if auc_ovo is not None:
                writer.add_scalar('info/{}_auc_ovo'.format(val_type), auc_ovo, epoch)
            writer.add_scalar('info/{}_f1'.format(val_type), f1, epoch)     

            metric_dir = Path(writer.log_dir).parent / val_type / 'epoch_{}'.format(epoch)
            write_metrics(
                metric_dir,
                metrics,
                metadata=benchmark_metadata(
                    algorithm.cfg,
                    method_display_name(algorithm.cfg),
                    val_type,
                    epoch,
                ),
            )
                
            logging.info('{} - epoch: {}, loss: {}, acc: {}, auc: {}, F1: {}.'.format
            (val_type, epoch, loss, acc, auc_ovo, f1))

    algorithm.train()
    return (-1.0 if auc_ovo is None else auc_ovo), loss
