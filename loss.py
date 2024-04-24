import torch
from torch import nn
import numpy as np

def compute_dist_euc(x1,x2,p1,p2):
    m, n = x1.shape[0], x2.shape[0]
    dist = torch.pow(x1, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(x2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist.addmm_(x1, x2.t(), beta=1, alpha=-2)
    dist = dist.clamp(min=1e-12).sqrt()
    mask = p1.expand(n, m).t().eq(p2.expand(m, n))
    return dist, mask

def compute_dist_kl(x1,x2,p1,p2):
    n1, c = x1.shape
    n2, c = x2.shape
    pr1 = x1.expand(n2, n1, c).detach()
    pr2 = x2.expand(n1, n2, c).detach()
    x1 = x1.clamp(1e-9).log().expand(n2, n1, c)
    x2 = x2.clamp(1e-9).log().expand(n1, n2, c)
    dist = (pr2 * x2.detach() - pr2 * x1.permute(1, 0, 2)).sum(dim=2) + \
           (pr1 * x1.detach() - pr1 * x2.permute(1, 0, 2)).sum(dim=2).t()
    mask = p1.expand(n2, n1).t().eq(p2.expand(n1, n2))
    return dist, mask

def compute_dist(x1,x2,p1,p2,d='euc'):
    if d == 'euc':
        return compute_dist_euc(x1, x2, p1, p2)
    if d == 'kl':
        return compute_dist_kl(x1, x2, p1, p2)

class hcc(nn.Module):
    def __init__(self, margin_euc=0.6, margin_kl=6.0):
        super(hcc, self).__init__()
        self.margin_euc = margin_euc
        self.margin_kl = margin_kl

    def forward(self, x, pids, d='euc'):
        if d == 'euc':
            margin = self.margin_euc
        if d == 'kl':
            margin = self.margin_kl
            x = x.softmax(dim=-1)
        p = len(pids.unique())
        c = x.shape[-1]
        pidhc = pids.reshape(2*p, -1)[:, 0]# pid编号
        hcen = x.reshape(2*p, -1, c).mean(dim=1)# 每个pid对应的中心，C维

        dist, mask = compute_dist(x, hcen, pids, pidhc, d)
        loss = []
        n, m = dist.shape
        for i in range(n // 2):
            loss.append(dist[i][m // 2:][mask[i][m // 2:]])
        for i in range(n // 2, n):
            loss.append(dist[i][:m // 2][mask[i][:m // 2]])
        loss1 = torch.cat(loss).mean()
        dist, mask = compute_dist(x, hcen, pids, pidhc, d)
        loss = []
        n, m = dist.shape
        for i in range(n):
            loss.append((margin - dist[i][mask[i] == 0]).clamp(0))
        loss2 = torch.cat(loss).mean()
        return loss1 + loss2

class ptcc(nn.Module):
    def __init__(self, margin_euc=0.6, margin_kl=6.0):
        super(ptcc, self).__init__()
        self.margin_euc = margin_euc
        self.margin_kl = margin_kl

    def forward(self, x, pids, d='euc'):
        if d == 'euc':
            margin = self.margin_euc
        if d == 'kl':
            margin = self.margin_kl
            x = x.softmax(dim=-1)
        p = len(pids.unique())
        c = x.shape[-1]
        pidhc = pids.reshape(2*p, -1)[:, 0]# pid编号
        hcen = x.reshape(2*p, -1, c).mean(dim=1)# 每个pid对应的中心，C维

        dist, mask = compute_dist(x, hcen, pids, pidhc, d)
        loss = []
        n, m = dist.shape
        for i in range(n // 2):
            loss.append(dist[i][m // 2:][mask[i][m // 2:]])
        for i in range(n // 2, n):
            loss.append(dist[i][:m // 2][mask[i][:m // 2]])
        loss = torch.cat(loss).mean()
        return loss

class cccc(nn.Module):
    def __init__(self, margin_euc=0.6, margin_kl=6.0):
        super(cccc, self).__init__()
        self.margin_euc = margin_euc
        self.margin_kl = margin_kl

    def forward(self, x, pids, d='euc'):
        if d == 'euc':
            margin = self.margin_euc
        if d == 'kl':
            margin = self.margin_kl
            x = x.softmax(dim=-1)
        p = len(pids.unique())
        c = x.shape[-1]
        pidhc = pids.reshape(2*p, -1)[:, 0]# pid编号
        hcen = x.reshape(2*p, -1, c).mean(dim=1)# 每个pid对应的中心，C维

        dist, mask = compute_dist(x, hcen, pids, pidhc, d)
        loss = []
        n, m = dist.shape
        for i in range(n // 2):
            loss.append(dist[i][:m // 2][mask[i][:m // 2]])
        for i in range(n // 2, n):
            loss.append(dist[i][m // 2:][mask[i][m // 2:]])
        loss = torch.cat(loss).mean()
        return loss

class triplet(nn.Module):
    def __init__(self, margin=0.6):
        super(triplet, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max())
            dist_an.append(dist[i][mask[i] == 0].min())
        dist_ap = torch.stack(dist_ap)
        dist_an = torch.stack(dist_an)

        # Compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        prec = dist_an.data > dist_ap.data
        length = torch.sqrt((inputs * inputs).sum(1)).mean()
        return loss

if __name__ == '__main__':
    pass



