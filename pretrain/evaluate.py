import logging
import os
import time

import torch
import torch.multiprocessing as mp
import torchvision.models as models
import  torchvision.datasets as datasets
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import destroy_process_group

import config as config
import utils as utils
import moco as moco
import data as data
import time
import pathlib
import builtins
import argparse


def get_parser():
    parser = argparse.ArgumentParser(prog='moco pretrain eval')
    parser.add_argument('path', type=str)
    parser.add_argument('pretrain', type=str)
    parser.add_argument('--arch', default='resnet50', type=str)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=16, type=int)
    
    parser.add_argument('--master_addr', default='localhost', type=str)
    parser.add_argument('--master_port', default='23333', type=str)
    parser.add_argument('--num_nodes', default=1, type=int)
    parser.add_argument('--this_node_id', default=0, type=int)
    parser.add_argument('--dist_backend', default='nccl', type=str)

    parser.add_argument('--output_dir', default='./output/', type=str)

    return parser


def merge_args(args1, args2):
    """
    用args1来更新args2
    """
    args2.path = args1.path
    args2.pretrain = args1.pretrain
    args2.arch = args1.arch
    args2.batch_size = args1.batch_size
    args2.num_workers = args1.num_workers
    args2.master_addr = args1.master_addr
    args2.master_port = args1.master_port
    args2.num_nodes = args1.num_nodes
    args2.this_node_id = args1.this_node_id
    args2.dist_backend = args1.dist_backend
    args2.output_dir = args1.output_dir

    return args2


def remove_prefix(sd, prefix):
    for key, value in sd.items():
        del sd[key]
        key = key.replace(prefix, "", 1)
        sd[key] = value
    return sd



def print_pass(msg):
    pass


def main_worker(local_rank, local_world_size, args):
    """
    用于训练的进程, 通常一个进程对应一个GPU

    Parameters
    ----------
    local_rank : 进程在本机的ID
    local_world_size : 本机共有多少进程
    """
    if local_rank != 0:
        time.sleep(1)
        builtins.print = print_pass

    # 加载预训练文件
    state_dict = torch.load(os.path.join(args.output_dir, 'pretrain', args.pretrain))
    args = merge_args(args, state_dict['args'])

    # 加载模型
    encoder = models.__dict__[args.arch]
    model = moco.MoCo(encoder, args.embed_dim, args.num_neg_samples, args.temp)
    model = model.cuda(local_rank)
    model.eval()
    loss = torch.nn.CrossEntropyLoss().cuda(local_rank)
    # 配置DDP
    utils.setup_ddp(local_rank, local_world_size, args)
    model = DDP(model, device_ids=[local_rank])
    model.load_state_dict(state_dict['model'])
    # 加载数据
    aug = data.TwoCropsWrapper(data.MoCoDataAugmentation())
    dataset = datasets.ImageFolder(root=args.path, transform=aug)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False,
                        pin_memory=True, num_workers=args.num_workers, drop_last=True)

    top1 = utils.AverageMeter("Acc@1", float)
    top5 = utils.AverageMeter("Acc@5", float)
    losses = utils.AverageMeter("Loss", float)
    
    # 进行每个epoch必要的设置
    sampler.set_epoch(0)

    for iter, (images, _) in enumerate(loader):
        query_image, key_image = images[0].cuda(local_rank), images[1].cuda(local_rank)
        
        # 获取模型的输出
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output, target = model(query_image, key_image)
            output = utils.concat_all_gather(output)
            target = utils.concat_all_gather(target)
            l = loss(output, target)

        # 在控制台输出信息
        acc1, acc5 = utils.accuracy(output, target, (1,5))
        top1.update(acc1)
        top5.update(acc5)
        losses.update(l.item())
    
        print(f"Iter-{iter}: {top1.average}\t{top5.average}\t{losses.average}")

    destroy_process_group()


if __name__ == '__main__':
    args = get_parser().parse_args()
    pathlib.Path(os.path.join(args.output_dir, "pretrain")).mkdir(exist_ok=True)

    # 开始训练
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    local_world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(local_world_size, args), nprocs=local_world_size,
                join=True)
