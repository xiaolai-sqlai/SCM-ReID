from __future__ import absolute_import

import math
import random
import copy
import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from einops import rearrange
from functools import reduce

from network.layer import BatchDrop, BatchErasing
from network.regnet_y import RegNetY, ConvX


class Attention(nn.Module):
    def __init__(self, num_part):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_part, 1))), requires_grad=True)

    def forward(self, q, k, v):
        logit_scale = torch.clamp(self.logit_scale, max=4.6052).exp()
        dots = q @ k.transpose(-2, -1) * logit_scale
        attn = dots.softmax(dim=-1)
        out = attn @ v
        return out


class SpectralClusteringMask(nn.Module):
    def __init__(self, in_planes, reduce_num=64, num_part=2, h=384, w=128, mask_part=1):
        super(SpectralClusteringMask, self).__init__()
        self.num_part = num_part
        self.h = h
        self.w = w
        self.mask_part = mask_part

        self.proj_in = ConvX(in_planes, reduce_num*num_part, groups=1, kernel_size=1, stride=1, act_type=None)
        self.att = Attention(num_part)
        self.proj_out = ConvX(reduce_num*num_part, num_part, groups=num_part, kernel_size=1, stride=1, act_type=None)

        self.learnable_masks = nn.Parameter(self.init_mask(), requires_grad=True)

    def init_mask(self):
        learnable_masks = []
        for i in range(self.num_part):
            chunks = self.num_part + self.mask_part - 1
            learnable_mask = torch.ones(1, 1, self.h, self.w)
            masks = torch.chunk(learnable_mask, dim=2, chunks=chunks)
            for j in range(self.mask_part):
                masks[i+j][:] = 0
            learnable_masks.append(torch.cat(masks, dim=2))

        return torch.cat(learnable_masks, dim=1)

    def forward(self, x):
        b, c, hh, ww = x.shape

        feat = self.proj_in(x)
        feat = rearrange(feat, 'b (h d) hh ww -> b (hh ww) h d', h=self.num_part)
        l2_feat = F.normalize(feat, dim=-1)
        out = self.att(l2_feat, l2_feat, feat)
        out = rearrange(out, 'b (hh ww) h d -> b (h d) hh ww', h=self.num_part, hh=hh, ww=ww)
        masks = torch.chunk((self.proj_out(out) * self.learnable_masks).clamp(0, 1), chunks=self.num_part, dim=1)
        return masks, sum(masks)


class GlobalAvgPool2d(nn.Module):
    def __init__(self, p=1):
        super(GlobalAvgPool2d, self).__init__()
        self.p = p
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = x.pow(self.p)
        out = self.gap(out)
        return out.pow(1/self.p)


class GlobalMaxPool2d(nn.Module):
    def __init__(self, p=1):
        super(GlobalMaxPool2d, self).__init__()
        self.p = p
        self.gap = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        out = x.pow(self.p)
        out = self.gap(out)
        return out.pow(1/self.p)


class SCM(nn.Module):
    def __init__(self, num_classes=751, num_parts=[1,2], feat_num=0, reduce_num=64, std=0.1, net="regnet_y_1_6gf", erasing=0.0, h=384, w=128, mask_part=1, kernel_size=1):
        super(SCM, self).__init__()
        self.num_parts = num_parts
        self.feat_num = feat_num
        self.kernel_size = kernel_size
        if self.training:
            self.erasing = nn.Identity()
            if erasing > 0:
                self.erasing = BatchErasing(smax=erasing)

        if net == "regnet_y_800mf":
            base = RegNetY(dims=[48,96,192,384], layers=[5,10,23,5], ratio=1.0, drop_path_rate=0.05)
            path = "pretrain/checkpoint_regnet_y_800mf.pth"
        elif net == "regnet_y_1_6gf":
            base = RegNetY(dims=[64,128,256,512], layers=[6,12,27,6], ratio=1.0, drop_path_rate=0.10)
            path = "pretrain/checkpoint_regnet_y_1_6gf.pth"
        elif net == "regnet_y_1_6gf_prelu":
            base = RegNetY(dims=[64,128,256,512], layers=[6,12,27,6], ratio=1.0, act_type="prelu", drop_path_rate=0.10)
            path = "pretrain/checkpoint_regnet_y_1_6gf_prelu.pth"

        old_checkpoint = torch.load(path)["state_dict"]
        new_checkpoint = dict()
        for key in old_checkpoint.keys():
            if key.startswith("module."):
                new_checkpoint[key[7:]] = old_checkpoint[key]
            else:
                new_checkpoint[key] = old_checkpoint[key]
        base.load_state_dict(new_checkpoint)

        self.stem = base.first_conv
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3

        base.layer4[0].main[1].conv.stride = (1, 1)
        base.layer4[0].skip[0].conv.stride = (1, 1)

        self.branch_1 = copy.deepcopy(nn.Sequential(base.layer4, base.head))
        self.branch_2 = copy.deepcopy(nn.Sequential(base.layer4, base.head))

        self.scm = SpectralClusteringMask(1024, reduce_num, num_parts[1], h, w, mask_part)
        self.pool_list = nn.ModuleList()
        self.feat_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.class_list = nn.ModuleList()

        for i in range(len(self.num_parts)):
            self.pool_list.append(GlobalAvgPool2d(p=1))
            if self.feat_num == 0:
                feat_num = 1024
                feat = nn.Identity()
            else:
                feat = nn.Linear(1024, feat_num, bias=False)
                init.kaiming_normal_(feat.weight)
            self.feat_list.append(feat)
            bn = nn.BatchNorm1d(feat_num)
            init.normal_(bn.weight, mean=1.0, std=std)
            init.normal_(bn.bias, mean=0.0, std=std)
            self.bn_list.append(bn)

            linear = nn.Linear(feat_num, num_classes, bias=False)
            init.normal_(linear.weight, std=0.001)
            self.class_list.append(linear)

        for i in range(sum(self.num_parts)):
            self.pool_list.append(GlobalMaxPool2d(p=1))
            if self.feat_num == 0:
                feat_num = 1024
                feat = nn.Identity()
            else:
                feat = nn.Linear(1024, feat_num, bias=False)
                init.kaiming_normal_(feat.weight)
            self.feat_list.append(feat)
            bn = nn.BatchNorm1d(feat_num)
            init.normal_(bn.weight, mean=1.0, std=std)
            init.normal_(bn.bias, mean=0.0, std=std)
            bn.bias.requires_grad = False
            self.bn_list.append(bn)

            linear = nn.Linear(feat_num, num_classes, bias=False)
            init.normal_(linear.weight, std=0.001)
            self.class_list.append(linear)


    def forward(self, x):
        if self.training:
            x = self.erasing(x)

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x1 = self.branch_1(x)
        x2 = self.branch_2(x)

        competitive_masks, cooperative_mask = self.scm(x2)
        x_chunk = [x1, x2 * cooperative_mask, x1]
        for competitive_mask in competitive_masks:
            x_chunk.append(x2 * competitive_mask)

        pool_list = []
        feat_list = []
        bn_list = []
        class_list = []

        for i in range(len(self.num_parts)+sum(self.num_parts)):
            pool = self.pool_list[i](x_chunk[i]).flatten(1)
            pool_list.append(pool)
            feat = self.feat_list[i](pool)
            feat_list.append(feat)
            bn = self.bn_list[i](feat)
            bn_list.append(bn)
            feat_class = self.class_list[i](bn)
            class_list.append(feat_class)

        if self.training:
            mask = 0
            for i in range(0, self.num_parts[1]):
                for j in range(i + 1, self.num_parts[1]):
                    mask = mask + F.max_pool2d(competitive_masks[i], kernel_size=self.kernel_size) * F.max_pool2d(competitive_masks[j], kernel_size=self.kernel_size)
            return class_list, bn_list[:2], mask / self.num_parts[1]
        return bn_list, competitive_masks, cooperative_mask


if __name__ == "__main__":
    base_1 = RegNetY(dims=[48,96,192,384], layers=[5,10,23,5], ratio=1.0, drop_path_rate=0.05)
    base_1.load_state(torch.load("pretrain/checkpoint_regnet_y_800mf.pth", "cpu"))
    base_1 = base_1.half()
    torch.save("checkpoint_regnet_y_800mf.pth", base_1.state_dict())

    base_2 = RegNetY(dims=[64,128,256,512], layers=[6,12,27,6], ratio=1.0, drop_path_rate=0.10)
    base_3 = RegNetY(dims=[64,128,256,512], layers=[6,12,27,6], ratio=1.0, act_type="prelu", drop_path_rate=0.10)
