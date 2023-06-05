import logging
import pathlib
import os

import torch
from torch.distributed import init_process_group



def setup_logging_system(args):
    pathlib.Path(args.output_dir).mkdir(exist_ok=True)
    path = os.path.join(args.output_dir, 'moco.log')
    logging.baseConfig(filename=path, encoding='utf-8', level=logging.INFO)


def setup_ddp(local_rank, local_world_size, args):
    global_world_size = local_world_size * args.num_nodes
    global_rank = local_rank + args.this_node_id * local_world_size
    init_process_group(backend=args.dist_backend,
                       rank=global_rank, world_size=global_world_size)


def update_key_encoder(model, args):
    m = args.key_encoder_momentum
    for param_q, param_k in zip(model.query_encoder, model.key_encoder):
        param_k.data.mul_(m)
        param_k.data.add_((1 - m) * param_q.data)


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    for milestone in args.schedule:
        if epoch >= milestone:
            lr *= 0.1
    for group in optimizer.param_group:
        group["lr"] = lr
