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

import config
import utils
import moco
import data


def main_worker(local_rank, local_world_size, args):
    """
    用于训练的进程, 通常一个进程对应一个GPU

    Parameters
    ----------
    local_rank : 进程在本机的ID
    local_world_size : 本机共有多少进程
    """
    # 定义模型
    encoder = models.__dict__[args.arch]
    model = moco.MoCo(encoder, args.embed_dim, args.num_neg_samples, args.temp, args.use_ddp)
    model = model.cuda(local_rank)
    loss = torch.nn.CrossEntropyLoss().cuda(local_rank)

    # 将模型使用DDP进行装饰
    if args.use_ddp:
        utils.setup_ddp(local_rank, local_world_size, args)
        model = DDP(model, device_ids=[local_rank])
        module = model.module
        # Run a dummy forward pass to initlize the parameters
        dummy_query = torch.rand(16, 3, 224, 224).cuda(local_rank)
        dummy_key = torch.rand(16, 3, 224, 224).cuda(local_rank)
        _, _ = module(dummy_query, dummy_key)
    else:
        module = model

    # 定义优化器
    optimizer = torch.optim.SGD(params=module.get_parameters(), momentum=args.momentum,
                                lr=args.lr)
    scalar = torch.cuda.amp.GradScalar()

    # 加载数据
    aug = data.TwoCropsWrapper(data.MoCoDataAugmentation())
    dataset = datasets.ImageFolder(root=args.path, transform=aug)
    sampler = DistributedSampler(dataset) if args.use_ddp else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False,
                        pin_memory=True, num_workers=args.num_workers, drop_last=True)
    
    # 检查是否使用check point
    mark = time.time()
    if args.use_ckp:
        assert args.use_ckp.startswith("checkpoint-")
        assert args.use_ckp.endswith(".pth")
        start_epoch = int(args.use_ckp.lstrip("checkpoint-").rstrip(".pth"))
        path = os.path.join(args.output_dir, args.use_ckp)
        module.load_state_dict(torch.load(path))
    else:
        start_epoch = 0

    
    # 开始训练
    for epoch in range(start_epoch, args.epochs):
        # 进行每个epoch必要的设置
        if args.use_ddp: sampler.set_epoch(epoch)
        utils.adjust_learning_rate(optimizer, epoch, args)

        for iter, (images, _) in enumerate(loader):
            query_image, key_image = images[0].cuda(local_rank), images[1].cuda(local_rank)
            
            # 获取模型的输出
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output, target = model(query_image, key_image)
                l = loss(output, target)

            # 更新query encoder
            optimizer.zero_grad()
            scalar.scale(l).backward()
            scalar.step(optimizer)
            scalar.update()

            # 更新key encoder
            utils.update_key_encoder(module, args)

            # 输出信息
            if local_rank == 0:
                image_s = args.batch_size / (time.time() - mark)
                print(f"Iter-{iter}\t{image_s} Images/s per GPU")
                mark = time.time()
            
        if epoch % args.save_ckp_freq == 0:
            path = os.path.join(args.output_dir, f'checkpoint-{epoch}.pth')
            torch.save(module.state_dict(), path)

        destroy_process_group()


if __name__ == '__main__':
    args = config.parser.parse_args()
    utils.setup_logging_system(args)

    # 检查DDP参数是否正确
    if (args.use_ddp and args.master_addr and args.master_port 
        and args.num_nodes and args.this_node_id):
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
