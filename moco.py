import torch
import torch.nn as nn
import torch.distributed as dist

from torch.distributed import all_gather
import utils


class MoCo(nn.Module):
    def __init__(self, encoder, embed_dim, num_neg_samples, temp, use_ddp):
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
        self.use_ddp = use_ddp

        # 设置neg samples队列
        self.register_buffer("queue", torch.rand(num_neg_samples, embed_dim))
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.queue = torch.nn.functional.normalize(self.queue, p=2, dim=-1)

        # 建模query encoder和key encoder
        self.query_encoder = encoder()
        self.key_encoder = encoder()
        self.query_encoder.fc = nn.Linear(self.query_encoder.fc.in_features, embed_dim)
        self.key_encoder.fc = nn.Linear(self.key_encoder.fc.in_features, embed_dim)

        # 加载模型参数
        for param_q, param_k in zip(self.query_encoder.parameters(), 
                                    self.key_encoder.parameters()):
            nn.init.xavier_uniform_(param_q.unsqueeze(0))
            param_k.requires_grad_(False)
            param_k.data.copy_(param_q.data)


    def get_parameters(self):
        return self.query_encoder.parameters()    
    

    def update_queue(self, key_feat):
        if self.use_ddp:
            key_feat = utils.concat_all_gather(key_feat)

        # 更新队列
        batch_size = key_feat.shape[0]
        assert self.queue.shape[0] % batch_size == 0
        ptr = int(self.ptr)
        self.queue[ptr: ptr + batch_size] = key_feat
        self.ptr[0] = (ptr + batch_size) % self.queue.shape[0]


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
            key_image, indices = utils.shuffle(key_image, self.use_ddp)
            key_feat = self.key_encoder(key_image)
            key_feat = utils.unshuffle(key_feat, indices, self.use_ddp)
            key_feat = torch.nn.functional.normalize(key_feat, p=2, dim=-1)
        
        # 计算相似度, 每个样本共有1个positive sample和K个negative samples
        l_pos = torch.einsum("nc,nc->n", [query_feat, key_feat]).unsqueeze(-1)
        l_neg = torch.einsum("nc,kc->nk", [query_feat, self.queue])
        logits = torch.cat([l_pos, l_neg], dim=-1) / self.temp

        # 期望分布
        target = torch.cat([torch.ones_like(l_pos), torch.zeros_like(l_neg)], dim=-1).to(logits.device)

        # 更新队列
        self.update_queue(key_feat)

        return logits, target
