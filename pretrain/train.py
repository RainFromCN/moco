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

    if args.use_ckp:
        # 加载checkpoint
        state_dict = torch.load(os.path.join(args.output_dir, "pretrain", args.use_ckp))
        # 加载模型、优化器、以及起始epoch和参数
        args = state_dict["args"]

    encoder = models.__dict__[args.arch]
    model = moco.MoCo(encoder, args.embed_dim, args.num_neg_samples, args.temp)
    model = model.cuda(local_rank)
    # 配置DDP
    utils.setup_ddp(local_rank, local_world_size, args)
    model = DDP(model, device_ids=[local_rank])
    # 定义优化器
    optimizer = torch.optim.SGD(params=model.module.get_parameters(),
                                momentum=args.momentum, lr=args.lr, weight_decay=args.weight_decay)

    # 检查是否使用check point
    if args.use_ckp:
        model.load_state_dict(state_dict["model"])
        optimizer.load_state_dict(state_dict["optimizer"])
        start_epoch = state_dict["epoch"]
        print(f"Use {args.use_ckp} to training. Start from epoch {start_epoch}")
    else:
        start_epoch = 0
        print("Training from scratch.")

    # 定义模型
    loss = torch.nn.CrossEntropyLoss().cuda(local_rank)
    scaler = torch.cuda.amp.GradScaler()

    # 加载数据
    aug = data.TwoCropsWrapper(data.MoCoDataAugmentation())
    dataset = datasets.ImageFolder(root=args.path, transform=aug)
    sampler = DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False,
                        pin_memory=True, num_workers=args.num_workers, drop_last=True)

    for epoch in range(start_epoch, args.epochs):
        top1 = utils.AverageMeter("Acc@1", float)
        top5 = utils.AverageMeter("Acc@5", float)
        losses = utils.AverageMeter("Loss", float)
        
        # 进行每个epoch必要的设置
        sampler.set_epoch(epoch)
        utils.adjust_learning_rate(optimizer, epoch, args)

        for iter, (images, _) in enumerate(loader):
            query_image, key_image = images[0].cuda(local_rank), images[1].cuda(local_rank)
            
            # 获取模型的输出
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output, target = model(query_image, key_image)
                l = loss(output, target)

            # 更新query encoder
            optimizer.zero_grad()
            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()

            # 更新key encoder
            utils.update_key_encoder(model.module, args)

            # 在控制台输出信息
            acc1, acc5 = utils.accuracy(output, target, (1,5))
            top1.update(acc1)
            top5.update(acc5)
            losses.update(l.item())
        
        print(f"Epoch[{epoch}/{args.epochs}]: {top1.average}\t{top5.average}\t{losses.average}")

        if epoch % args.save_ckp_freq == 0:
            # 保存checkpoint
            path = os.path.join(args.output_dir, 'pretrain', f'checkpoint-{epoch}.pth')
            utils.save_checkpoint(path, model.state_dict(), optimizer.state_dict(), epoch, args)

    destroy_process_group()


if __name__ == '__main__':
    args = config.parser.parse_args()
    pathlib.Path(os.path.join(args.output_dir, "pretrain")).mkdir(exist_ok=True)

    # 开始训练
    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port
    local_world_size = torch.cuda.device_count()
    mp.spawn(main_worker, args=(local_world_size, args), nprocs=local_world_size,
                join=True)
