import os
import argparse
import pathlib
import torch
import torchvision.models as models
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch import autocast
import torch.multiprocessing as mp
import utils


class Head(torch.nn.Module):
    def __init__(self, embed_dim, num_cls):
        self.linear = torch.nn.Linear(embed_dim, num_cls)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        torch.nn.init.constant_(self.linear.bias.data, 0)

    def forward(self, x):
        return self.linear(x)
    

class Augmentation:
    def __init__(self):
        self.aug = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __call__(self, x):
        return self.aug(x)


def config_parser():
    parser = argparse.ArgumentParser("classify")

    # MISC
    parser.add_argument("path", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("--output_dir", default='./output', type=str)
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--save_ckp_freq", default=1, type=int)

    # 模型
    parser.add_argument("--arch", default="resnet50", type=str)
    parser.add_argument("--embed_dim", default=128, type=int)
    parser.add_argument("--num_cls", default=1000, type=int)

    # 训练
    parser.add_argument("--lr", default=30, type=float)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=10, type=int)
    
    # 分布式
    parser.add_argument("--use_ddp", default=True, type=bool)
    parser.add_argument("--master_addr", default="localhost", type=str)
    parser.add_argument("--master_port", default="23333", type=str)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--this_node_id", default=0, type=int)
    parser.add_argument("--dist_backend", default="nccl", type=str)

    return parser


def main_worker(local_rank, local_world_size, args):

    # 加载backbone
    assert args.model.startswith('checkpoint-') and args.model.endswith('.pth')
    module = models.__dict__[args.arch]
    module.fc = torch.nn.Linear(in_features=module.fc.in_features, out_features=args.embed_dim)
    for param in module.parameters():
        param.requires_grad_(False)

    # 加载预训练参数
    path = os.path.join(args.output_dir, "pretrain", args.model)
    state_dict = torch.load(path, map_location='cpu')
    for name, param in module.named_parameters():
        param.data.copy_(state_dict[f"key_encoder.{name}"])
    for name, buffer in module.named_buffers():
        buffer.data.copy_(state_dict[f"key_encoder.{name}"])

    # 加载映射头并迁移至GPU
    module = Head(args.embed_dim, args.num_cls)(module)
    module = module.cuda(local_rank)
    loss = torch.nn.CrossEntropyLoss().cuda(local_rank)

    # 加载优化器
    optimizer  = torch.optim.SGD(module.parameters() + loss.parameters(), lr=args.lr, momentum=args.momentum,
                                 weight_decay=args.weight_decay)
    scaler = GradScaler()
    
    # 加载数据集
    dataset = ImageFolder(root=args.path, transform=Augmentation())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True if args.use_ddp else False,
                        sampler=DistributedSampler(dataset) if args.use_ddp else None, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True)
    
    # 配置DDP
    if args.use_ddp:
        utils.setup_ddp(local_rank, local_world_size, args)
        utils.print("Use ddp for training.")
        model = torch.nn.parallel.DistributedDataParallel(module, device_ids=[local_rank])
    else:
        utils.print(f"Use single GPU for training.")
        model = module


    # 开始训练
    for epoch in range(args.epochs):
        for i, (data, label) in enumerate(loader):
            data, label = data.cuda(local_rank), label.cuda(local_rank)

            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(data)
                l = loss(logits, label)

            scaler.scale(l).backward()
            scaler.step(optimizer)
            scaler.update()

            utils.print(f"Iter-{i}\tLoss-{l.item():.5f}")

        if epoch % args.save_ckp_freq == 0:
            path = os.path.join(args.output_dir, "classify", "checkpoint-{epoch}.pth")
            torch.save(module, path)

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()
    pathlib.Path(os.path.join(args.output_dir, "classify")).mkdir(exist_ok=True)

    os.environ["MASTER_ADDR"] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    if args.use_ddp:
        local_world_size = torch.cuda.device_count()
        mp.spawn(main_worker, args=(local_world_size, args), nprocs=local_world_size, join=True)
    else:
        main_worker(0, 1, args)

    utils.print("Training process is finished!")
