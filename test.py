import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_model_params
from neuralop import LpLoss, H1Loss
import torch.multiprocessing as mp

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from fairscale.nn.data_parallel import FullyShardedDataParallel

import torch

import argparse

def benchmark(rank, args, world_size):
    print(rank)
    device = 'cuda'
    
    RPC_PORT = 29501
    init_method_pgroup = "tcp://localhost:{}".format(RPC_PORT)
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size, init_method=init_method_pgroup
    )
    torch.cuda.set_device(rank)
    
    train_loader, test_loaders, data_processor = load_darcy_flow_small(
            n_train=1000, batch_size=32,
            test_resolutions=[16, 32], n_tests=[100, 50],
            test_batch_sizes=[32, 32],
            positional_encoding=True
    )
    data_processor = data_processor.to(device)
    
    model = TFNO(n_modes=(32, 32), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42, n_layers=64)
    model = model.to(device)
    model = FullyShardedDataParallel(model)
    
    n_params = count_model_params(model)
    print(f'\nOur model has {n_params} parameters.')
    sys.stdout.flush()
    
    optimizer = torch.optim.Adam(model.parameters(),
                                    lr=8e-3,
                                    weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    
    train_loss = h1loss
    eval_losses={'h1': h1loss, 'l2': l2loss}
    
    print('\n### MODEL ###\n', model)
    print('\n### OPTIMIZER ###\n', optimizer)
    print('\n### SCHEDULER ###\n', scheduler)
    print('\n### LOSSES ###')
    print(f'\n * Train: {train_loss}')
    print(f'\n * Test: {eval_losses}')
    sys.stdout.flush()
    
    trainer = Trainer(model=model, n_epochs=20,
                      device=device,
                      data_processor=data_processor,
                      wandb_log=False,
                      log_test_interval=3,
                      use_distributed=False,
                      verbose=True)
    
    trainer.train(train_loader=train_loader,
                  test_loaders=test_loaders,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  regularizer=False,
                  training_loss=train_loss,
                  eval_losses=eval_losses)
    
    test_samples = test_loaders[32].dataset
    
    print("Peak allocated bytes on cuda:{}: {:4f}GB".format(
                dist.get_rank(), torch.cuda.memory_stats(dist.get_rank())["allocated_bytes.all.peak"] / 2**30
            )
        )

    break

def parse_args():
    parser = argparse.ArgumentParser(description="benchmark")
    parser.add_argument("--max_batch", type=int, default=4, help="Max number of batches")
    
if __name__ == "__main__":
    args = parse_args()
    print(f"Running DP benchmark with args: {args}")
    num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(torch.cuda.device_count())
    mp.spawn(
        benchmark,
        args=(args, num_devices),
        nprocs=num_devices,
        join=True,
    )
