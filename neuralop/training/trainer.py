import torch
from torch.cuda import amp
from timeit import default_timer
import pathlib

from .callbacks import PipelineCallback
import neuralop.mpu.comm as comm
from neuralop.losses import LpLoss
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
    dim_size[0] = dim_size[0] * world_size

    output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
    torch.distributed.all_gather_into_tensor(
        output, input_.contiguous(), group=group
    )
    tensor_list = output.chunk(world_size, dim=0)
    output = torch.cat(tensor_list, dim=-2).contiguous()

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

class Trainer:
    def __init__(self, *, 
                 model, 
                 n_epochs, 
                 wandb_log=True, 
                 device=None, 
                 amp_autocast=False,
                 data_processor=None,
                 callbacks = None,
                 log_test_interval=1, 
                 log_output=False, 
                 use_distributed=False, 
                 verbose=False):
        """
        A general Trainer class to train neural-operators on given datasets

        Parameters
        ----------
        model : nn.Module
        n_epochs : int
        wandb_log : bool, default is True
        device : torch.device
        amp_autocast : bool, default is False
        data_processor : class to transform data, default is None
            if not None, data from the loaders is transform first with data_processor.preprocess,
            then after getting an output from the model, that is transformed with data_processor.postprocess.
        log_test_interval : int, default is 1
            how frequently to print updates
        log_output : bool, default is False
            if True, and if wandb_log is also True, log output images to wandb
        use_distributed : bool, default is False
            whether to use DDP
        verbose : bool, default is False
        """

        if callbacks:
            assert type(callbacks) == list, "Callbacks must be a list of Callback objects"
            self.callbacks = PipelineCallback(callbacks=callbacks)
            self.override_load_to_device = (self.callbacks.device_load_callback_idx is not None)
            self.overrides_loss = self.callbacks.overrides_loss
        else:
            self.callbacks = []
            self.override_load_to_device = False
            self.overrides_loss = False
        
        if verbose:
            print(f"{self.override_load_to_device=}")
            print(f"{self.overrides_loss=}")

        if self.callbacks:
            self.callbacks.on_init_start(model=model, 
                 n_epochs=n_epochs, 
                 wandb_log=wandb_log, 
                 device=device, 
                 amp_autocast=amp_autocast, 
                 log_test_interval=log_test_interval, 
                 log_output=log_output, 
                 use_distributed=use_distributed, 
                 verbose=verbose)

        self.model = model
        self.n_epochs = n_epochs

        self.wandb_log = wandb_log
        self.log_test_interval = log_test_interval
        self.log_output = log_output
        self.verbose = verbose
        self.use_distributed = use_distributed
        self.device = device
        self.amp_autocast = amp_autocast
        self.data_processor = data_processor

        if self.callbacks:
            self.callbacks.on_init_end(model=model, 
                 n_epochs=n_epochs, 
                 wandb_log=wandb_log, 
                 device=device, 
                 amp_autocast=amp_autocast, 
                 log_test_interval=log_test_interval, 
                 log_output=log_output, 
                 use_distributed=use_distributed, 
                 verbose=verbose)
        
    def train(self, train_loader, test_loaders,
            optimizer, scheduler, regularizer,
              training_loss=None, eval_losses=None):
        
        """Trains the given model on the given datasets.
        params:
        train_loader: torch.utils.data.DataLoader
            training dataloader
        test_loaders: dict[torch.utils.data.DataLoader]
            testing dataloaders
        optimizer: torch.optim.Optimizer
            optimizer to use during training
        optimizer: torch.optim.lr_scheduler
            learning rate scheduler to use during training
        training_loss: training.losses function
            cost function to minimize
        eval_losses: dict[Loss]
            dict of losses to use in self.eval()
        """

        if self.callbacks:
            self.callbacks.on_train_start(train_loader=train_loader, test_loaders=test_loaders,
                                    optimizer=optimizer, scheduler=scheduler, 
                                    regularizer=regularizer, training_loss=training_loss, 
                                    eval_losses=eval_losses)
            
        if training_loss is None:
            training_loss = LpLoss(d=2)

        if eval_losses is None: # By default just evaluate on the training loss
            eval_losses = dict(l2=training_loss)

        errors = None

        
        try:
            group = torch.distributed.distributed_c10d._get_default_group()
            rank = torch.distributed.get_rank(group=group)
            world_size = torch.distributed.get_world_size(group=group)
        except:
            rank = 0
            world_size = 1

        for epoch in range(self.n_epochs):

            if self.callbacks:
                self.callbacks.on_epoch_start(epoch=epoch)

            avg_loss = 0
            avg_lasso_loss = 0
            self.model.train()
            t1 = default_timer()
            train_err = 0.0

            for idx, sample in enumerate(train_loader):
                
                
                if self.callbacks:
                    self.callbacks.on_batch_start(idx=idx, sample=sample)

                optimizer.zero_grad(set_to_none=True)
                if regularizer:
                    regularizer.reset()

                if self.data_processor is not None:
                    sample = self.data_processor.preprocess(sample)
                else:
                    # load data to device if no preprocessor exists
                    sample = {k:v.to(self.device) for k,v in sample.items() if torch.is_tensor(v)}

                
                # sample['x'] = sample['x'].repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)
                # sample['y'] = sample['y'].repeat_interleave(8, dim=2).repeat_interleave(8, dim=3)
                # print(sample['x'].shape, sample['y'].shape)
                
                b, c, h, w = sample['x'].size()
                chunk_size = math.ceil(h / world_size)
                sample['x'] = list(sample['x'].split(chunk_size, dim=2))[rank]
                # sample['y'] = list(sample['y'].split(chunk_size, dim=2))[rank]

                if self.amp_autocast:
                    with amp.autocast(enabled=True):
                        out  = self.model(**sample)
                else:
                    out  = self.model(**sample)

                out = _GatherFromModelParallelRegion.apply(out)

                # print(out[0][0], out.shape)
                # exit()

                if self.data_processor is not None:
                    out, sample = self.data_processor.postprocess(out, sample)

                if self.callbacks:
                    self.callbacks.on_before_loss(out=out)

                loss = 0.

                if self.overrides_loss:
                    if isinstance(out, torch.Tensor):
                        loss += self.callbacks.compute_training_loss(out=out.float(), **sample, amp_autocast=self.amp_autocast)
                    elif isinstance(out, dict):
                        loss += self.callbacks.compute_training_loss(**out, **sample, amp_autocast=self.amp_autocast)
                else:
                    if self.amp_autocast:
                        with amp.autocast(enabled=True):
                            if isinstance(out, torch.Tensor):
                                loss = training_loss(out.float(), **sample)
                            elif isinstance(out, dict):
                                loss += training_loss(**out, **sample)
                    else:
                        if isinstance(out, torch.Tensor):
                            loss = training_loss(out.float(), **sample)
                        elif isinstance(out, dict):
                            loss += training_loss(**out, **sample)
                
                if regularizer:
                    loss += regularizer.loss

                # torch.distributed.all_reduce(loss, group=group)
                # loss.div_(world_size)
                # print(loss)
                # exit()
                
                loss.backward()
                del out

                # for param_name, param in self.model.named_parameters():
                #     if rank == 0:
                #         print(param_name, param.shape, param.grad.view(-1)[0])
                # exit()
                optimizer.step()
                train_err += loss.item()
        
                with torch.no_grad():
                    avg_loss += loss.item()
                    if regularizer:
                        avg_lasso_loss += regularizer.loss

                if self.callbacks:
                    self.callbacks.on_batch_end()

            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(train_err)
            else:
                scheduler.step()

            epoch_train_time = default_timer() - t1            

            train_err /= len(train_loader)
            avg_loss  /= self.n_epochs
            
            if epoch % self.log_test_interval == 0: 

                if self.callbacks:
                    self.callbacks.on_before_val(epoch=epoch, train_err=train_err, time=epoch_train_time, \
                                           avg_loss=avg_loss, avg_lasso_loss=avg_lasso_loss)
                

                for loader_name, loader in test_loaders.items():
                    errors = self.evaluate(eval_losses, loader, log_prefix=loader_name)

                if self.callbacks:
                    self.callbacks.on_val_end()
            
            if self.callbacks:
                self.callbacks.on_epoch_end(epoch=epoch, train_err=train_err, avg_loss=avg_loss)

        return errors

    def evaluate(self, loss_dict, data_loader,
                 log_prefix=''):
        """Evaluates the model on a dictionary of losses
        
        Parameters
        ----------
        loss_dict : dict of functions 
          each function takes as input a tuple (prediction, ground_truth)
          and returns the corresponding loss
        data_loader : data_loader to evaluate on
        log_prefix : str, default is ''
            if not '', used as prefix in output dictionary

        Returns
        -------
        errors : dict
            dict[f'{log_prefix}_{loss_name}] = loss for loss in loss_dict
        """

        if self.callbacks:
            self.callbacks.on_val_epoch_start(log_prefix=log_prefix, loss_dict = loss_dict, data_loader=data_loader)

        self.model.eval()

        errors = {f'{log_prefix}_{loss_name}':0 for loss_name in loss_dict.keys()}

        n_samples = 0
        with torch.no_grad():
            for idx, sample in enumerate(data_loader):

                n_samples += sample['y'].size(0)
                if self.callbacks:
                    self.callbacks.on_val_batch_start(idx=idx, sample=sample)

                if self.data_processor is not None:
                    sample = self.data_processor.preprocess(sample)
                else:
                    # load data to device if no preprocessor exists
                    sample = {k:v.to(self.device) for k,v in sample.items() if torch.is_tensor(v)}
                    
                out = self.model(**sample)

                if self.data_processor is not None:
                    out, sample = self.data_processor.postprocess(out, sample)

                if self.callbacks:
                    self.callbacks.on_before_val_loss(out=out)
                
                for loss_name, loss in loss_dict.items():
                    if self.overrides_loss:
                        if isinstance(out, torch.Tensor):
                            val_loss = self.callbacks.compute_training_loss(out.float(), **sample)
                        elif isinstance(out, dict):
                            val_loss = self.callbacks.compute_training_loss(**out, **sample)
                    else:
                        if isinstance(out, torch.Tensor):
                            val_loss = loss(out, **sample)
                        elif isinstance(out, dict):
                            val_loss = loss(out, **sample)
                        if val_loss.shape == ():
                            val_loss = val_loss.item()

                    errors[f'{log_prefix}_{loss_name}'] += val_loss

                if self.callbacks:
                    self.callbacks.on_val_batch_end()
    
        for key in errors.keys():
            errors[key] /= n_samples
        
        if self.callbacks:
            self.callbacks.on_val_epoch_end(errors=errors, sample=sample, out=out)
        
        del out

        return errors

