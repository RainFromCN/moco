import torch
import torch.nn as nn
import torch.distributed as dist

from torch.distributed import all_gather


class MoCo(nn.Module):
    def __init__(self, encoder, embed_dim, num_neg_samples, temp):
        """
        MoCo

        Parameters
        ----------
        encoder : nn.Module. 骨干网络, 用于提取特征.
        embed_dim : int. 特征维度
        num_neg_samples : int. 负样本的数量, 必须是batch-size的整数倍.
        """
        super().__init__()
        self.temp = temp

        # 设置neg samples队列
        self.register_buffer("queue", torch.rand(num_neg_samples, embed_dim))
        self.register_buffer("ptr", torch.zeros(1))
        self.queue = torch.nn.functional.normalize(self.queue, p=2, dim=-1)

        # 建模query encoder和key encoder
        self.query_encoder = encoder()
        self.key_encoder = encoder()
        self.query_encoder.fc = nn.LazyLinear(embed_dim)
        self.key_encoder.fc = nn.LazyLinear(embed_dim)

        # 加载模型参数
        for param_q, param_k in zip(self.query_encoder, self.key_encoder):
            param_q.data.copy_(param_k.data)


    def get_parameters(self):
        return self.query_encoder.parameters()
    

    def update_queue(self, key_feat):
        # 获取进程总数
        global_world_size = torch.distributed.get_world_size()
        # 开辟缓存区
        tensor_list = [torch.zeros_like(key_feat) for _ in range(global_world_size)]
        # 汇聚同类项
        all_gather(tensor_list, key_feat)
        key_feat_gather = torch.cat(tensor_list, dim=0)
        global_batch_size = key_feat_gather.shape[0]
        # 更新队列
        assert self.queue.shape[0] % global_batch_size
        ptr = int(self.ptr)
        self.queue[ptr: ptr + global_batch_size] = key_feat_gather
        self.ptr[0] = (ptr + global_batch_size) % self.queue.shape[0]


    def forward(self, query_image, key_image):
        """
        Parameters
        ----------
        query_image : M, C, H, W
        key_image : M, C, H, W

        Returns
        -------
        logits : 没有过softmax的分布
        target : one-hot向量
        """
        # 获取特征表示
        query_feat = self.query_encoder(query_image)
        query_feat = torch.nn.functional.normalize(query_feat, p=2, dim=-1)
        with torch.no_grad():
            key_feat = self.key_encoder(key_image)
            key_feat = torch.nn.functional.normalize(key_feat, p=2, dim=-1)

        self.update_queue(key_feat)
        
        # 计算相似度, 每个样本共有1个positive sample和K个negative samples
        l_pos = torch.einsum("nc,nc->n", [query_feat, key_feat]).unsqueeze(-1)
        l_neg = torch.einsum("nc,kc->nk", [query_feat, self.queue])
        logits = torch.cat([l_pos, l_neg], dim=-1) / self.temp

        # 期望分布
        target = torch.cat([torch.ones_like(l_pos), torch.zeros_like(l_neg)], dim=-1)

        return logits, target
