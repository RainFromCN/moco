import logging
import pathlib
import os

import torch
from torch.distributed import init_process_group, get_world_size, all_gather, get_rank, broadcast



def setup_logging_system(args):
    pathlib.Path(args.output_dir).mkdir(exist_ok=True)
    path = os.path.join(args.output_dir, 'moco.log')
    logging.basicConfig(filename=path, level=logging.INFO)


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
def shuffle(x, use_ddp):
    if use_ddp:
        x_gather = concat_all_gather(x)
        indices = torch.randperm(x_gather.shape[0]).to(x.device)
        broadcast(indices, src=0)
        start = get_rank() * x.shape[0]
        local_indices = indices[start: start + x.shape[0]]
        return x_gather[local_indices], indices
    else:
        indices = torch.randperm(x.shape[0]).to(x.device)
        return x[indices], indices


@torch.no_grad()
def unshuffle(x, indices, use_ddp):
    if use_ddp:
        x_gather = concat_all_gather(x)
        start = get_rank() * x.shape[0]
        reverse_indices = torch.argsort(indices)
        local_reverse_indices = reverse_indices[start: start + x.shape[0]]
        return x_gather[local_reverse_indices]
    else:
        return x[torch.argsort(indices)]
