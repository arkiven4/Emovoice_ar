import os
import json
import argparse
import math
from tqdm import tqdm
import numpy as np
import time

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.sampler import WeightedRandomSampler

import bitsandbytes as bnb
from data_utils import TextMelLoader, TextMelCollate, DistributedBucketSampler
import utils

from hparams import create_hparams
from models import OverFlow

global_step = 0
global_tqdm = None


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "65530"

    hparams = create_hparams()
    mp.spawn(
        train_and_eval,
        nprocs=n_gpus,
        args=(
            n_gpus,
            hparams,
        ),
    )


def train_and_eval(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=hps.model_dir)
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(
        backend="nccl", init_method="env://", world_size=n_gpus, rank=rank
    )
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    # Weight Sampler
    with open(hps.data.training_files, "r") as txt_file:
        lines = txt_file.readlines()

    attr_names_samples = np.array([item.split("|")[1] for item in lines])
    unique_attr_names = np.unique(attr_names_samples).tolist()
    attr_idx = [unique_attr_names.index(l) for l in attr_names_samples]
    attr_count = np.array(
        [len(np.where(attr_names_samples == l)[0]) for l in unique_attr_names]
    )
    weight_attr = 1.0 / attr_count
    dataset_samples_weight = np.array([weight_attr[l] for l in attr_idx])
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(
        dataset_samples_weight
    )
    weights, attr_names, attr_weights = (
        torch.from_numpy(dataset_samples_weight).float(),
        unique_attr_names,
        np.unique(dataset_samples_weight).tolist(),
    )
    weights = weights * 1.0
    w_sampler = WeightedRandomSampler(weights, len(weights))

    train_dataset = TextMelLoader(hps.validation_files, hps, hps.normaliser)
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True,
        weights=w_sampler,
    )
    collate_fn = TextMelCollate(hps.n_frames_per_step)
    train_loader = DataLoader(
        train_dataset,
        num_workers=8,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn,
        batch_sampler=train_sampler,
    )

    if rank == 0:
        # self.hparams.validation_files, self.hparams, [self.hparams.normaliser]
        val_dataset = TextMelLoader(hps.validation_files, hps, hps.normaliser)
        val_loader = DataLoader(
            val_dataset,
            num_workers=8,
            shuffle=False,
            batch_size=hps.train.batch_size,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    model = OverFlow(hps).cuda(rank)

    optimizer = bnb.optim.AdamW(
        model.parameters(),
        hps.train.learning_rate,
        betas=hps.train.betas,
        eps=hps.train.eps,
    )

    model = DDP(model, device_ids=[rank])
    epoch_str = 1
    global_step = 0

    start_epoch = 0
    total_iter = 0

    start_iter = 0

    scaler = GradScaler(enabled=hps.train.fp16_run)

    if hps.train.warm_start:
        model = utils.warm_start_model(hps.train.warm_start_checkpoint, model, hps.train.ignored_layer)
    else:
        try:
            utils.load_checkpoint(model, optimizer, scaler, start_epoch, start_iter, hps.train.fp16_run, utils.latest_checkpoint_path(hps.model_dir, "overflow_*.pt"))
        except Exception as e:
            print("Not Found Checkpoint")
            print(e)

    model.train()
    torch.cuda.synchronize()

    for epoch in range(start_epoch, hps.train.epochs + 1):
        epoch_start_time = time.perf_counter()
        
        if hps.distributed_run:
            train_loader.sampler.set_epoch(epoch)
        
        model.train()

        accumulated_steps = 0
        num_iters = len(train_loader)
        for batch_idx, batch in train_loader:
            if accumulated_steps == 0:
                if epoch_iter == num_iters:
                    break

                total_iter += 1
                epoch_iter += 1
                iter_start_time = time.perf_counter()

                adjust_learning_rate(total_iter, optimizer, hps.train.learning_rate, hps.train.warmup_steps)
                model.zero_grad()

            x, y = model.parse_batch(batch)
            with autocast(enabled=hps.train.fp16_run):
                log_probs = model(x)
                loss = -log_probs.mean()

            if hps.train.fp16_run:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if hps.train.distributed_run:
                reduced_loss = reduce_tensor(loss.data, hps.train.num_gpus).item()
            else:
                reduced_loss = loss.item()

            if np.isnan(reduced_loss):
                raise Exception("loss is NaN")

            accumulated_steps += 1
            iter_loss += reduced_loss

            if accumulated_steps % 1 == 0:
                if hps.train.fp16_run:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hps.train.grad_clip_thresh)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hps.train.grad_clip_thresh)
                    optimizer.step()

                iter_time = time.perf_counter() - iter_start_time
                epoch_loss += iter_loss

            if rank == 0:
                if batch_idx % hps.log_interval == 0:
                    (mel_output, state_travelled, input_parameters, output_parameters) = model.sample(x[0][0], x[1][0], langs=x[5][0], speakers=x[6][0], emos=x[7][0])

                    scalar_dict = {
                        "loss/g/total": loss,
                        "learning_rate": optimizer.param_groups[0]["lr"],
                        "grad_norm": grad_norm,
                        "iter_time": iter_time,
                    }

                    utils.summarize(
                        writer=writer,
                        global_step=global_step,
                        images={
                            "mel_output": utils.plot_spectrogram_to_numpy(mel_output.data.cpu().numpy()),
                            "mel_original": utils.plot_spectrogram_to_numpy(y[0].data.cpu().numpy()),
                            "0_attn": utils.plot_alpha_scaled_to_numpy(torch.exp(model.hmm.log_alpha_scaled[0, :, :]).T),
                        },
                        scalars=scalar_dict,
                    )

                accumulated_steps = 0
                iter_loss = 0

        # Finished epoch
        epoch_time = time.perf_counter() - epoch_start_time
        model.eval()

        tik = time.perf_counter()
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                x, y = model.parse_batch(batch)
                log_probs = model(x)
                loss = -log_probs.mean()

        val_took = time.perf_counter() - tik

        scalar_dict = {
            "loss/g/total": loss,
            "iter_time": val_took,
        }

        utils.summarize(
            writer=writer_eval,
            global_step=global_step,
            scalars=scalar_dict,
        )

        print("====> Epoch: {}".format(epoch))
        maybe_save_checkpoint(rank, model, optimizer, scaler, epoch, total_iter, hps, hps)
        logger.flush()
    # Finished training

def adjust_learning_rate(total_iter, opt, learning_rate, warmup_iters=None):
    if warmup_iters == 0:
        scale = 1.0
    elif total_iter > warmup_iters:
        scale = 1. / (total_iter ** 0.5)
    else:
        scale = total_iter / (warmup_iters ** 1.5)

    for param_group in opt.param_groups:
        param_group['lr'] = learning_rate * scale


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt.true_divide(num_gpus)

def maybe_save_checkpoint(rank, model, optimizer, scaler, epoch, total_iter, config, hparams):
    if rank != 0:
        return

    intermediate = (hparams.train.epochs_per_checkpoint > 0 and epoch % hparams.train.epochs_per_checkpoint == 0)
    if not intermediate and epoch < hparams.train.epochs:
        return

    fpath = os.path.join(hparams.model_dir, f"fastpitch_{epoch}.pt")
    print(f"Saving model and optimizer state at epoch {epoch} to {fpath}")
    checkpoint = {'epoch': epoch,
                  'iteration': total_iter,
                  'config': config,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()}
    
    if hparams.train.fp16_run:
        checkpoint['scaler'] = scaler.state_dict()

    torch.save(checkpoint, fpath)