import logging
import pathlib
import os

import torch
from torch.distributed import init_process_group, get_world_size, all_gather, get_rank, broadcast
import builtins


def save_checkpoint(file_path, model, optimizer, epoch, args):
    state_dict = {
        "model": model,
        "optimizer": optimizer,
        "epoch": epoch,
        "args": args,
    }
    torch.save(state_dict, file_path)


def setup_ddp(local_rank, local_world_size, args):
    global_world_size = local_world_size * args.num_nodes
    global_rank = local_rank + args.this_node_id * local_world_size
    init_process_group(backend=args.dist_backend,
                       rank=global_rank, world_size=global_world_size)


@torch.no_grad()
def update_key_encoder(model, args):
    m = args.key_encoder_momentum
    for param_q, param_k in zip(model.query_encoder.parameters(), 
                                model.key_encoder.parameters()):
        param_k.data.mul_(m)
        param_k.data.add_((1 - m) * param_q.data)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.schedule:
        if epoch >= milestone:
            lr *= 0.1
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def concat_all_gather(x):
    x_list = [torch.zeros_like(x) for _ in range(get_world_size())]
    all_gather(x_list, x, async_op=False)
    return torch.cat(x_list, dim=0)


@torch.no_grad()
def shuffle(x):
    x_gather = concat_all_gather(x)
    indices = torch.randperm(x_gather.shape[0]).to(x.device)
    broadcast(indices, src=0)
    start = get_rank() * x.shape[0]
    local_indices = indices[start: start + x.shape[0]]
    return x_gather[local_indices], indices


@torch.no_grad()
def unshuffle(x, indices):
    x_gather = concat_all_gather(x)
    start = get_rank() * x.shape[0]
    reverse_indices = torch.argsort(indices)
    local_reverse_indices = reverse_indices[start: start + x.shape[0]]
    return x_gather[local_reverse_indices]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, dim=-1, largest=True, sorted=True)
        target = target.unsqueeze(-1).expand_as(pred)
        table = pred == target
        res = []
        for k in topk:
            acc = table[:, :k].sum().item() / output.shape[0]
            res.append(acc * 100)
        return res
        


class AverageMeter:
    def __init__(self, name, type=float):
        self.name = name
        self.type = type
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
        self.avg = self.sum / self.count

    @property
    def average(self):
        return f"{self.name}-{self.avg:.4f}" if self.type is float else f"{self.name}-{self.avg}"

    def __str__(self):
        return f"{self.name}-{self.val:.4f}" if self.type is float else f"{self.name}-{self.val}"
    