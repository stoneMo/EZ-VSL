import torch
import torch.nn as nn
import torch.nn.functional as F


def contra_loss(logits):
    bsz = logits.shape[0]
    labels = torch.zeros(bsz).long().to(logits.device)
    loss = F.cross_entropy(logits, labels)
    return loss


def bce_loss(S):
    # S: (bsz, bsz, H, W)
    bsz = S.shape[0]
    labels = torch.eye(bsz).float().cuda()
    mask_ii = torch.eye(bsz).long().cuda()
    # print("mask_ii:", mask_ii)

    mask_ij = torch.logical_not(mask_ii).long().cuda()

    S_logits = torch.max(S.reshape(bsz, bsz, -1),dim=-1)[0]/0.07

    loss_fn = nn.BCELoss(reduction='sum')
    
    loss = 1/mask_ii.sum() * loss_fn(S_logits*mask_ii, labels*mask_ii) + 1/mask_ij.sum() * loss_fn(S_logits*mask_ij, labels*mask_ij)

    return loss


def ce_loss(S):
    bsz = S.shape[0]
    labels = torch.arange(bsz).long().to(S.device)
    if S.ndim == 4:
        S = S.flatten(-2, -1).max(dim=-1)[0]
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fn(S, labels) + loss_fn(S.permute(1, 0), labels)
    return loss


def ce2_loss(S):
    bsz = S.shape[0]
    height, width = S.shape[-2:]
    labels = torch.arange(bsz).long().to(S.device)
    if S.ndim == 4:
        S = torch.topk(S.flatten(-2, -1), dim=-1, sorted=False, k=int(height*width*0.3))[0].mean(-1)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fn(S, labels) + loss_fn(S.permute(1, 0), labels)
    return loss


def ce3_loss(S):
    bsz = S.shape[0]
    labels = torch.arange(bsz).long().to(S.device)
    if S.ndim == 4:
        S = S.flatten(-2, -1).mean(dim=-1)
    loss_fn = nn.CrossEntropyLoss(reduction='mean')
    loss = loss_fn(S, labels) + loss_fn(S.permute(1, 0), labels)
    return loss


def std_loss(embed_img, threshold):
    # embed_vis: (bsz, C, H, W)
    bsz, channel, height, width = embed_img.shape
    embed_img = embed_img.view(bsz, channel, height*width)
    std = torch.std(embed_img, dim=-1).mean(1)
    threshold = torch.ones_like(std) * threshold
    zeros = torch.zeros_like(std)
    loss = torch.max(-std + threshold, zeros).mean()
    return loss


def std2_loss(embed_img, threshold):
    # embed_vis: (bsz, C, H, W)
    bsz, channel, height, width = embed_img.shape
    embed_img = embed_img.view(bsz, channel, height*width)
    embed_img = F.normalize(embed_img, dim=1) * (embed_img.shape[1]**0.5)
    std = torch.std(embed_img, dim=-1).mean(1)
    threshold = torch.ones_like(std) * threshold
    zeros = torch.zeros_like(std)
    loss = torch.max(-std + threshold, zeros).mean()
    return loss






# # Sum up over spatial dimensions
# Pos = torch.sum(Pos, dim=(1, 2))
# Neg = torch.sum(Neg, dim=(1, 2))

# # Optimization objective
# bsize = Pos.shape[0]
# loss = -1/bsize * torch.sum(torch.log(torch.exp(Pos) / (torch.exp(Pos) + torch.exp(Neg))))
if __name__ == '__main__':

    bsz = 3
    H, W = 5, 5
    # S = torch.rand((bsz, bsz, H, W))
    S = torch.zeros((bsz, bsz, H, W))

    print("S:", S)

    loss_ce = bce_loss(nn.Sigmoid()(S))

    print("loss_ce:", loss_ce)
