import os
import argparse
import pathlib
import torch
import torchvision.models as models
import torch.nn as nn
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.cuda.amp import GradScaler
from torch import autocast
import torch.multiprocessing as mp
import utils


class Classifer(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.backbone = models.__dict__[args.arch]()
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, args.num_cls)

        # 加载backbone的预训练参数
        path = os.path.join(args.output_dir, "pretrain", args.pretrain)
        assert os.path.exists(path)
        state_dict = torch.load(path, map_location='cpu')
        for key, value in list(state_dict.items()):
            if key.startswith("query_encoder") and not key.startswith("query_encoder.fc"):
                state_dict[key.replace("query_encoder.", "", 1)] = value
            del state_dict[key]
        msg = self.backbone.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        
        # 初始化全连接层
        path = os.path.join(args.output_dir, "classify", args.parameters)
        assert os.path.exists(path)
        for param1, param2 in zip(self.backbone.fc.parameters(), torch.load(path, map_location='cpu').parameters()):
            param1.data.copy_(param2)
    
    def forward(self, x):
        return self.backbone(x)
    

class Augmentation:
    def __init__(self):
        self.aug = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

    def __call__(self, x):
        return self.aug(x)


def config_parser():
    parser = argparse.ArgumentParser("classify")

    # MISC
    parser.add_argument("path", type=str)
    parser.add_argument("pretrain", type=str)
    parser.add_argument("parameters", type=str)
    parser.add_argument("--output_dir", default='./output', type=str)
    parser.add_argument("--num_workers", default=16, type=int)

    # 模型
    parser.add_argument("--arch", default="resnet50", type=str)
    parser.add_argument("--num_cls", default=1000, type=int)

    # 预测
    parser.add_argument("--batch_size", default=256, type=int)
    
    # 分布式
    parser.add_argument("--use_ddp", default=False, type=bool)
    parser.add_argument("--master_addr", default="localhost", type=str)
    parser.add_argument("--master_port", default="23333", type=str)
    parser.add_argument("--num_nodes", default=1, type=int)
    parser.add_argument("--this_node_id", default=0, type=int)
    parser.add_argument("--dist_backend", default="nccl", type=str)

    return parser


def main_worker(local_rank, local_world_size, args):
    top1 = utils.AverageMeter("Acc@1", float)
    top5 = utils.AverageMeter("Acc@5", float)
    losses = utils.AverageMeter("Loss", float)

    # 加载模型
    module = Classifer(args).cuda(local_rank)
    loss = torch.nn.CrossEntropyLoss()
    module.eval()
    
    # 加载数据集
    dataset = ImageFolder(root=args.path, transform=Augmentation())
    sampler = DistributedSampler(dataset) if args.use_ddp else None
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True if args.use_ddp else False,
                        sampler=DistributedSampler(dataset) if args.use_ddp else None, num_workers=args.num_workers,
                        pin_memory=True, drop_last=True)
    
    # 配置DDP
    if args.use_ddp:
        utils.setup_ddp(local_rank, local_world_size, args)
        utils.print("Use ddp for training.", args.use_ddp)
        model = torch.nn.parallel.DistributedDataParallel(module, device_ids=[local_rank])
    else:
        utils.print("Use single GPU for training.", args.use_ddp)
        model = module

    # 开始测试
    if args.use_ddp:
        sampler.set_epoch(0)

    with torch.no_grad():
        for i, (data, label) in enumerate(loader):
            data, label = data.cuda(local_rank), label.cuda(local_rank)

            with autocast(device_type='cuda', dtype=torch.float16):
                logits = model(data)
                l = loss(logits, label)

            acc1, acc5 = utils.accuracy(logits, label, topk=(1,5))
            print(f"Iter-{i}\t{top1}\t{top5}")
            top1.update(acc1)
            top5.update(acc5)
            losses.update(l.item())

    print(f"{top1.average}\t{top5.average}\t{losses.average}")

    if args.use_ddp:
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

    utils.print("Evaluate process is finished!", args.use_ddp)
