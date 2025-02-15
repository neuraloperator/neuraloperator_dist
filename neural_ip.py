import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys
import time
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
import numpy as np

import argparse
import math

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, "{} is not divisible by {}".format(numerator, denominator)
    
def divide(numerator, denominator):
    """Ensure that numerator is divisible by the denominator and return
    the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator
    
def split_tensor(
    tensor: torch.Tensor, num_partitions: int, contiguous_split_chunks: bool = False, dim: int = -1
) -> List[torch.Tensor]:
    """ Split a tensor along its last dimension.

        Arguments:
            tensor: input tensor.
            num_partitions: number of partitions to split the tensor
            contiguous_split_chunks: If True, make each chunk contiguous
                                     in memory.

        Returns:
            A list of Tensors
    """
    # Get the size and dimension.
    dim_size = divide(tensor.size()[dim], num_partitions)
    # Split.
    tensor_list = torch.split(tensor, dim_size, dim=dim)
    # Note: torch.split does not create contiguous tensors by default.
    if contiguous_split_chunks:
        return tuple(chunk.contiguous() for chunk in tensor_list)

    return tensor_list

def _split_along_last_dim(input_):
    """Split the tensor along its first dimension and keep the
    corresponding slice."""
    group = torch.distributed.distributed_c10d._get_default_group()
    world_size = torch.distributed.get_world_size(group=group)
    rank = torch.distributed.get_rank(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    input_list = split_tensor(input_, world_size, dim=-2)

    output = input_list[rank].contiguous()

    return output

def _gather_along_last_dim(input_):
    """Gather tensors and concatinate along the first dimension."""

    group = torch.distributed.distributed_c10d._get_default_group()
    world_size = torch.distributed.get_world_size(group=group)
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    dim_size[-2] = dim_size[-2] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed._all_gather_base(
        output, input_.contiguous(), group=group
    )

    return output
    
class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from model parallel region and concatinate."""

    @staticmethod
    def symbolic(graph, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def forward(ctx, input_):
        return _gather_along_last_dim(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _split_along_last_dim(grad_output)

def init_random_seed(seed: int):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    
def benchmark(rank, args, world_size):
    print(rank)
    device = 'cuda'

    init_random_seed(42)
    
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
    
    model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64, factorization='tucker', rank=0.42)
    # model = TFNO(n_modes=(128, 128), hidden_channels=256, projection_channels=512, factorization='tucker', rank=0.42)
    model = model.to(device)

    optimizer_dict = {}
    # def optimizer_hook2(parameter) -> None:
    #     print('-------------------', optimizer_dict[parameter], parameter.shape, parameter.grad.view(-1)[0])
        
    def optimizer_hook(parameter) -> None:
        dist.all_reduce(parameter.grad)
        # print('-------------------', optimizer_dict[parameter], parameter.shape, parameter.grad.view(-1)[0])
    
    for param_name, param in model.named_parameters():
        print(param_name, param.shape)
        optimizer_dict[param] = param_name

        param.register_post_accumulate_grad_hook(optimizer_hook)
    
        # if not 'fno_blocks' in param_name:
        #     print('no-------------------',param_name)
        #     param.register_post_accumulate_grad_hook(optimizer_hook)
        # else:
        #     param.register_post_accumulate_grad_hook(optimizer_hook2)
            

    print(model)
    # model = DDP(model)
    
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
    
    trainer = Trainer(model=model, n_epochs=100,
                      device=device,
                      data_processor=data_processor,
                      wandb_log=False,
                      log_test_interval=3,
                      use_distributed=True,
                      verbose=True)
    
    epoch_start_time = time.time()
    trainer.train(train_loader=train_loader,
                  test_loaders=test_loaders,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  regularizer=False,
                  training_loss=train_loss,
                  eval_losses=eval_losses)
    wps = 100 / (time.time() - epoch_start_time)
    print("Throughput(wps) is {:.2f}.".format(wps))
    
    test_samples = test_loaders[32].dataset
    
    print("Peak allocated bytes on cuda:{}: {:4f}GB".format(
                dist.get_rank(), torch.cuda.memory_stats(dist.get_rank())["allocated_bytes.all.peak"] / 2**30
            )
        )

    test_samples = test_loaders[32].dataset

    fig = plt.figure(figsize=(7, 7))
    for index in range(3):
        data = test_samples[index]
        data = data_processor.preprocess(data, batched=False)
        # Input x
        x = data['x']
        # Ground-truth
        y = data['y']
        # Model prediction

        print(x.shape)
        c, h, w = x.size()
        chunk_size = math.ceil(h / world_size)
        x = list(x.split(chunk_size, dim=1))[rank]
        out = model(x.unsqueeze(0))
        out = _GatherFromModelParallelRegion.apply(out)

        ax = fig.add_subplot(3, 3, index*3 + 1)
        ax.imshow(x[0].cpu(), cmap='gray')
        if index == 0:
            ax.set_title('Input x')
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(3, 3, index*3 + 2)
        ax.imshow(y.cpu().squeeze())
        if index == 0:
            ax.set_title('Ground-truth y')
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(3, 3, index*3 + 3)
        ax.imshow(out.cpu().squeeze().detach().numpy())
        if index == 0:
            ax.set_title('Model prediction')
        plt.xticks([], [])
        plt.yticks([], [])

    fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
    plt.tight_layout()
    if rank == 0:
        fig.savefig('my_figure.png')

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
