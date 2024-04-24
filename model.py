from torchvision import models
import torch, math
import torch.nn as nn
from torch.nn import init
from functools import partial

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)

    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)
        m.bias.requires_grad_(False)

class resnet(nn.Module):
    def __init__(self, class_num):
        super(resnet, self).__init__()
        model = models.resnet50(pretrained=True)
        for mo in model.layer4[0].modules():
            if isinstance(mo, nn.Conv2d):
                mo.stride = (1, 1)
        self.layer = nn.Sequential(*list(model.children())[:-2])
        self.dim = 2048
        self.np, self.nh = 6, 8
        #--------------------------------------------------------------------------
        # self.mam = MAM(2048)
        # num_patches = 18 * 9
        # depth = 1
        # self.cls_token = nn.ParameterList([nn.Parameter(torch.zeros(1, 1, dim))])
        # trunc_normal_(self.cls_token[0], std=.02)

        # self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, num_patches + 1, dim))])
        # trunc_normal_(self.pos_embed[0], std=.02)
        
        # self.pos_embed = nn.ParameterList([nn.Parameter(torch.zeros(1, num_patches, dim))])
        # trunc_normal_(self.pos_embed[0], std=.02)

        # dpr = [x.item() for x in torch.linspace(0, 0.1, depth)] # dpr = [x.item() for x in torch.linspace(0, 0.1, 1)]
        # self.blocks = nn.ModuleList([
        #     Block(
        #         dim=2048, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None,
        #         drop=0., attn_drop=0., drop_path=dpr[i], norm_layer=partial(nn.LayerNorm, eps=1e-6))
        #     for i in range(depth)])
        # self.norm = partial(nn.LayerNorm, eps=1e-6)(dim)
        #--------------------------------------------------------------------------
        self.bottleneck = nn.BatchNorm1d(self.dim)
        self.bottleneck.apply(weights_init)
        self.classifier = nn.Linear(self.dim, class_num, bias=False)
        self.classifier.apply(weights_init)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.ignored_params = nn.ModuleList([self.bottleneck, self.classifier])

    def forward(self, x, camids=None, pids=None):
        x = self.layer(x)

        #--------------------------------------------------------------------------
        # x = self.mam(x)
        # B, x_ = x.shape[0], x # B, C, H, W
        # if pids is not None:
        #     print (pids)
        # # exit(0)
        # # if camids is not None:
        # #     is_rgb = ((camids != 3) * (camids != 6))
        # x_ = x_.flatten(2).transpose(1, 2) # [64, 162, 2048] B, P, C
        # # cls_tokens = self.cls_token[0].expand(B, -1, -1)
        # # print (cls_tokens)
        # # x_ = torch.cat((cls_tokens, x_), dim=1)
        # x_ = x_ + self.pos_embed[0] # [64, 163, 2048]
        # for blk in self.blocks:
        #     x_ = blk(x_)
        # x_ = self.norm(x_)
        
        # pool = x_.mean(dim=1)
        # feat = self.bottleneck(pool)
        # if self.training:
        #     y = self.classifier(feat)
        #     return pool, y
        # else:
        #     return pool, feat
        #--------------------------------------------------------------------------
        B, C, H, W = x.shape
        pp = x.view(B, self.nh, self.dim//self.nh, self.np, H//self.np, W)
        pp = pp.mean(-1).mean(-1).permute(0, 1, 3, 2).contiguous()
        pp = pp.view(B, self.nh*self.np, self.dim//self.nh)

        pool = self.avgpool(x).squeeze()
        feat = self.bottleneck(pool)
        if self.training:
            y = self.classifier(feat)
            return pool, y, pp
        else:
            return pool, feat

















class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

if __name__ == '__main__':
    pass





