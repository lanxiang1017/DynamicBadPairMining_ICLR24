from __future__ import print_function
import os
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class InfoNCE(nn.Module):
    def __init__(self, temperature, device):
        super(InfoNCE, self).__init__()
        self.device = device
        self.temperature = temperature

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        mask = mask.bool()
        return mask

    def forward(self, out_1, out_2):
        """
            assume out_1 and out_2 are normalized
            out_1: [batch_size, dim]
            out_2: [batch_size, dim]
        """
        # gather representations in case of distributed training
        bsz = out_1.shape[0]

        out_1 = F.normalize(out_1, dim=1)
        out_2 = F.normalize(out_2, dim=1)

        out_1_dist = out_1
        out_2_dist = out_2

        out = torch.cat([out_1, out_2], dim=0)
        out_dist = torch.cat([out_1_dist, out_2_dist], dim=0)

        similarity = torch.exp(torch.mm(out, out_dist.t()) / self.temperature)

        neg_mask = self.mask_correlated_samples(bsz).to(self.device)

        neg = torch.sum(similarity * neg_mask, 1)

        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / self.temperature)

        pos = torch.cat([pos, pos], dim=0)

        logits = -torch.log(pos / (pos + neg))
        logits_re = logits.view(2, -1)
        pos_loss = torch.mean(logits_re, dim=0)

        return pos_loss

def loss_ntxent(features, device):
    fn = InfoNCE(temperature=0.2, device=device)
    pos_loss = fn(features[0], features[1])

    return pos_loss
