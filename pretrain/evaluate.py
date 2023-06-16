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

import pretrain.config as config
import pretrain.utils as utils
import pretrain.moco as moco
import pretrain.data as data
import time
import pathlib
import argparse


def get_parser():
    parser = argparse.ArgumentParser("moco pretrain evaluator")

    # 必选参数
    parser.add_argument("path", type=str)

    # 关于模型结构
    parser.add_argument("--arch", default="resnet50")
    parser.add_argument("--embed_dim", default=128, type=int)
    parser.add_argument("--num_neg_samples", default=65536, type=int)
    parser.add_argument("--temp", default=0.07, type=float)

    # 关于训练
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--num_workers", default=16, type=int)

    # 关于分布式训练
    parser.add_argument("--use_ddp", default=False, type=bool)
    parser.add_argument("--master_addr", default="localhost", type=str)
    parser.add_argument("--master_port", default="23333", type=str)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--this_node_id", default=0, type=int)
    parser.add_argument("--dist_backend", default="nccl", type=str)    

    return parser



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
    else:
        utils.setup_logging_system(args)
    
    # 定义模型
    encoder = models.__dict__[args.arch]
    model = moco.MoCo(encoder, args.embed_dim, args.num_neg_samples, args.temp, args.use_ddp)
    model = model.cuda(local_rank)
    model.eval()
    loss = torch.nn.CrossEntropyLoss().cuda(local_rank)
    loss.eval()

    # 将模型使用DDP进行装饰
    if args.use_ddp:
        utils.setup_ddp(local_rank, local_world_size, args)
        model = DDP(model, device_ids=[local_rank])

    # 加载数据
    aug = data.TwoCropsWrapper(data.MoCoDataAugmentation())
    dataset = datasets.ImageFolder(root=args.path, transform=aug)
    sampler = DistributedSampler(dataset) if args.use_ddp else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, 
                        shuffle=False if args.use_ddp else True,
                        pin_memory=True, num_workers=args.num_workers, drop_last=True)
    
    # 开始训练
    if local_rank == 0:
        top1 = utils.AverageMeter("Acc@1", float)
        top5 = utils.AverageMeter("Acc@5", float)
        losses = utils.AverageMeter("Loss", float)

    
    for epoch in range(args.epochs):
        # 进行每个epoch必要的设置
        if args.use_ddp: sampler.set_epoch(epoch)

        for images, _ in loader:
            query_image, key_image = images[0].cuda(local_rank), images[1].cuda(local_rank)
            
            # 获取模型的输出
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output, target = model(query_image, key_image)
                l = loss(output, target)

            # 在控制台输出信息
            if local_rank == 0:
                acc1, acc5 = utils.accuracy(output, target, (1,5))
                top1.update(acc1)
                top5.update(acc5)
                losses.update(l.item())
        
        if local_rank == 0:
            # 保存epoch的训练日志
            message = f"Epoch[{epoch}/{args.epochs}]: {top1.average}\t{top5.average}\t{losses.average}\t"
            utils.print(message)
            logging.info(message)

    destroy_process_group()


if __name__ == '__main__':
    args = get_parser().parse_args()
    pathlib.Path(os.path.join(args.output_dir, "pretrain")).mkdir(exist_ok=True)
    utils.setup_logging_system(args)

    # 检查DDP参数是否正确
    if (args.use_ddp and args.master_addr and args.master_port 
        and args.num_nodes is not None and args.this_node_id is not None):
        logging.info("Use ddp for training.")
    else:
        args.use_ddp = False
        logging.info("Use single gpu for training")

    # 开始训练
    if args.use_ddp:
        os.environ["MASTER_ADDR"] = args.master_addr
        os.environ["MASTER_PORT"] = args.master_port
        local_world_size = torch.cuda.device_count()
        mp.spawn(main_worker, args=(local_world_size, args), nprocs=local_world_size,
                 join=True)
    else:
        main_worker(local_rank=0, local_world_size=1, args=args)    
