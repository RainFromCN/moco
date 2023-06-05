import argparse

import torchvision.models as models


parser = argparse.ArgumentParser(prog="moco")

models_name = [
    name for name in models.__dict__
    if name.islower() and not name.startswith('__')
]


# 必选参数
parser.add_argument("path", type=str)

# 关于模型结构
parser.add_argument("--arch", default="resnet50", choices=models_name)
parser.add_argument("--embed_dim", default=128, type=int)
parser.add_argument("--num_neg_samples", default=65536, type=int)
parser.add_argument("--temp", default=0.07, type=float)

# 关于训练
parser.add_argument("--batch_size", default=256, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--lr", default=0.03, type=float)
parser.add_argument("--schedule", nargs='*', default=[120, 160], type=int)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--key_encoder_momentum", default=0.999, type=float)
parser.add_argument("--num_workers", default=16, type=int)

# 关于分布式训练
parser.add_argument("--use_ddp", default=False, type=bool)
parser.add_argument("--master_addr", default=None, type=str)
parser.add_argument("--master_port", default=None, type=int)
parser.add_argument("--num_nodes", default=None, type=int)
parser.add_argument("--this_node_id", default=None, type=int)
parser.add_argument("--dist_backend", default="nccl", type=str)

# MISC
parser.add_argument("--output_dir", default="./output", type=str)
