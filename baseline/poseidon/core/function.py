import time
import os
import numpy as np
import torch

from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from utils.common import TRAIN_PHASE, VAL_PHASE, TEST_PHASE
from datasets.process.heatmaps_process import get_final_preds, get_final_preds_coor
from utils.utils_save_results import save_batch_examples
from .evaludate import accuracy, pck_accuracy

def train_batch_accumulation(cfg, train_loader, model, criterion, optimizer, epoch, output_dir, device, experiment_dir, save_examples=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # Switch to train mode
    model.train()
    model.set_phase(TRAIN_PHASE)

    end = time.time()
    total_time = time.time()

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', mininterval=5.0, leave=False)

    scaler = GradScaler()

    # Define the number of accumulation steps
    accumulation_steps = cfg.TRAIN.ACCUMULATION_STEPS  # Example: 32 steps to simulate a larger batch size

    # Initialize gradient accumulation variables
    optimizer.zero_grad()  # Zero the gradients before starting

    for i, (x, meta, target_heatmaps, target_heatmaps_weight) in enumerate(pbar):
        data_start = time.time()

        # Move input and target data to device
        x = x.to(device, non_blocking=True)
        target_heatmaps = target_heatmaps.to(device, non_blocking=True)
        target_heatmaps_weight = target_heatmaps_weight.to(device, non_blocking=True)
        
        data_time.update(time.time() - data_start)

        # Enable automatic mixed precision (AMP) for efficient training
        with autocast():
            output = model(x, meta)
            loss = criterion(output, target_heatmaps, target_heatmaps_weight)
            loss = loss / accumulation_steps  # Normalize loss by accumulation steps

        # Backpropagate the loss and accumulate gradients
        scaler.scale(loss).backward()

        # Accumulate gradients and perform optimizer step after accumulation_steps
        if (i + 1) % accumulation_steps == 0:
            # Perform optimizer step and update scaler
            scaler.step(optimizer)
            scaler.update()

            # Zero the gradients for the next set of accumulation steps
            optimizer.zero_grad()

        # Compute accuracy
        _, avg_acc, cnt, _ = accuracy(output.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())

        # Update running loss and accuracy
        losses.update(loss.item() * accumulation_steps, x.size(0))  # Multiply back by accumulation steps to get the actual loss
        acc.update(avg_acc, cnt)

        # Update timing
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            tqdm_desc = f'Epoch {epoch} [Loss: {losses.avg:.6f}] [Acc: {acc.avg:.3f}] [Data: {data_time.avg:.3f}s] [Batch: {batch_time.avg:.3f}s]'
            pbar.set_description(tqdm_desc)


    # Print summary for the epoch
    print(f'Training Epoch {epoch} Summary:\t'
          f'Loss {losses.avg:.6f}\t'
          f'Acc {acc.avg:.3f}\t')
    


    total_time = time.time() - total_time
    print("\033[95m" + f"Total time: {total_time // 60:.0f} minutes and {total_time % 60:.2f} seconds" + "\033[0m\n")

    return losses.avg, acc.avg

def train(cfg, train_loader, model, criterion, optimizer, epoch, output_dir, device, experiment_dir, save_examples=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()
    model.set_phase(TRAIN_PHASE)

    end = time.time()
    total_time = time.time()

    # Save some examples with keypoints only if save_examples flag is True
    #if save_examples:
        #input_prev, input_x, input_next, meta, _ , _ = next(iter(train_loader))
        #save_batch_examples(input_x, meta['target'], experiment_dir, epoch)

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', mininterval=5.0, leave=False)

    scaler = GradScaler()

    #for i, (input_prev, input_x, input_next, meta, target_heatmaps, target_heatmaps_weight) in enumerate(pbar):
    for i, (x, meta, target_heatmaps, target_heatmaps_weight) in enumerate(pbar):
        
        data_start = time.time()

        #x = torch.stack([input_prev, input_x, input_next], dim=1).to(device, non_blocking=True
        x = x.to(device, non_blocking=True)
        target_heatmaps = target_heatmaps.to(device, non_blocking=True)
        target_heatmaps_weight = target_heatmaps_weight.to(device, non_blocking=True)
        
        data_time.update(time.time() - data_start)

        with autocast():
            output = model(x, meta)
            #loss = criterion(output, meta['target'].to(device), meta['target_weight'].to(device))
            loss = criterion(output, target_heatmaps, target_heatmaps_weight)

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        _, avg_acc, cnt, _ = accuracy(output.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())

        losses.update(loss.item(), x.size(0))

        acc.update(avg_acc, cnt)

        # acc.update(output.acc.item(), len(input_x))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            tqdm_desc = f'Epoch {epoch} [Loss: {losses.avg:.6f}] [Acc: {acc.avg:.3f}] [Data: {data_time.avg:.3f}s] [Batch: {batch_time.avg:.3f}s]'
            pbar.set_description(tqdm_desc)


    print(f'Training Epoch {epoch} Summary:\t'
          f'Loss {losses.avg:.6f}\t'
          f'Acc {acc.avg:.3f}')

    total_time = time.time() - total_time
    print("\033[95m" + f"Total time: {total_time // 60:.0f} minutes and {total_time % 60:.2f} seconds" + "\033[0m\n")

    return losses.avg, acc.avg

def validate(config, val_loader, val_dataset, model, criterion, output_dir, epoch, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()
    model.set_phase(VAL_PHASE)

    num_samples = len(val_loader.dataset)
    all_preds = np.zeros((num_samples, config.MODEL.NUM_JOINTS, 3), dtype=np.float32)
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    filenames_map = {}
    filenames_counter = 0
    imgnums = []
    idx = 0

    pbar = tqdm(val_loader, desc=f'Epoch {epoch}', mininterval=5.0, leave=False)

    with torch.no_grad():
        end = time.time()
        #for i, (input_prev, input_x, input_next, meta, target_heatmaps, target_heatmaps_weight) in enumerate(pbar):
        for i, (x, meta, target_heatmaps, target_heatmaps_weight) in enumerate(pbar):

            #x = torch.stack([input_prev, input_x, input_next], dim=1).to(device, non_blocking=True)

            x = x.to(device, non_blocking=True)
            target_heatmaps = target_heatmaps.to(device, non_blocking=True)
            target_heatmaps_weight = target_heatmaps_weight.to(device, non_blocking=True)

            with autocast():
                output = model(x, meta)
                #pred_coor = output.pred_jts.detach().cpu().numpy()
                #score_coor = output.maxvals.detach().cpu().numpy()

                loss = criterion(output, target_heatmaps, target_heatmaps_weight)

            _, avg_acc, cnt, _ = accuracy(output.detach().cpu().numpy(), target_heatmaps.detach().cpu().numpy())
            
            # get len batch size
            acc.update(avg_acc, cnt)
            losses.update(loss.item(), x.size(0))

            # for evaluation
            for ff in range(len(meta['image'])):
                cur_nm = meta['image'][ff]
                if cur_nm not in filenames_map:
                    filenames_map[cur_nm] = [filenames_counter]
                else:
                    filenames_map[cur_nm].append(filenames_counter)
                filenames_counter += 1

            center = meta['center'].numpy()
            scale = meta['scale'].numpy()
            score = meta['score'].numpy()
            num_images =  x.size(0)

            preds, maxvals = get_final_preds(output.clone().cpu().numpy(), center, scale)
            #preds, maxvals = get_final_preds(output, center, scale)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            all_boxes[idx:idx + num_images, 0:2] = center[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = scale[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(scale * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])
            idx += num_images

            # self._after_val_iter(meta["image"], preds, maxvals, config.OUTPUT_DIR, center, scale)

            if i % 100 == 0:
                tqdm_desc = f'Epoch {epoch} [Loss: {losses.avg:.6f}] [Acc: {acc.avg:.3f}]'
                pbar.set_description(tqdm_desc)

            

    name_values, perf_indicator = val_dataset.evaluate(config, all_preds, config.OUTPUT_DIR, all_boxes,
                                                       filenames_map, filenames, imgnums)

    print(f'Validation Summary:\t'
          f'Epoch {epoch}\t'
          f'Loss {losses.avg:.6f}\t'
          f'Acc {acc.avg:.3f}\t'
          f'Ap Mean or PCK {perf_indicator:.3f}')

    total_time = time.time() - end
    # print total time in minutes and seconds in purple color
    print("\033[95m" + f"Total time: {total_time // 60:.0f} minutes and {total_time % 60:.2f} seconds" + "\033[0m\n")

    return name_values, perf_indicator, losses.avg, acc.avg  # AP mean

def _after_val_iter(self, image, preds_joints, preds_confidence, vis_output_dir, center, scale):
    cfg = self.cfg
    # prepare data
    coords = np.concatenate([preds_joints, preds_confidence], axis=-1)
    bboxes = []
    for index in range(len(center)):
        xyxy_bbox = cs2box(center[index], scale[index], pattern="xyxy")
        bboxes.append(xyxy_bbox)

    if cfg.DEBUG.VIS_SKELETON or cfg.DEBUG.VIS_BBOX:
        from .vis_helper import draw_skeleton_in_origin_image
        draw_skeleton_in_origin_image(image, coords, bboxes, vis_output_dir, vis_skeleton=cfg.DEBUG.VIS_SKELETON,
                                        vis_bbox=cfg.DEBUG.VIS_BBOX)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0