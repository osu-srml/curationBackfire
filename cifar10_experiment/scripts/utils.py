import os
import json
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import CosineAnnealingLR

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from transformers import CLIPModel, CLIPProcessor

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def entropy(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    probs = probs.clamp(min=eps)
    return -(probs * probs.log()).sum(dim=-1)

CIFAR_MEAN = torch.tensor([0.4914, 0.4822, 0.4465], device=torch.device).view(1, 3, 1, 1)
CIFAR_STD = torch.tensor([0.2470, 0.2435, 0.2616], device=torch.device).view(1, 3, 1, 1)


# reward helpers
@torch.no_grad()
def _gap_stats(vec_mk: torch.Tensor) -> dict:
    """
    vec_mk: [M, K] CPU tensor
    returns mean gap, median gap, mean(top1 - mean), mean(std)
    """
    vmax = vec_mk.max(dim=1).values
    vmin = vec_mk.min(dim=1).values
    vmean = vec_mk.mean(dim=1)
    vstd = vec_mk.std(dim=1, unbiased=False)

    gap = vmax - vmin  # [M]
    top1_minus_mean = vmax - vmean

    med = torch.quantile(gap, 0.5)
    return {
        'gap_mean': float(gap.mean().item()),
        'gap_median': float(med.item()),
        'top1_minus_mean': float(top1_minus_mean.mean().item()),
        'std_mean': float(vstd.mean().item()),
    }
    
@torch.no_grad()
def summarize_candidate_diffs(
    x_cand_cpu: torch.Tensor,  # [M*K,3,32,32] in [-1, 1] CPU
    M: int,
    K: int, 
    score_vec_cpu: torch.Tensor,  # [M*K] CPU, the reward used for curation
    tag: str,
    tau_select: float=None,  # if use BT, pass tau to log entropy
) -> dict:
    """
    Summarize candidate diversity along warm/cool/realism and the actual selections score.
    Returns a dict to print/log.
    """
    assert x_cand_cpu.size(0) == M * K
    assert score_vec_cpu.numel() == M * K

    warm, cool, realism = _color_realism_terms(x_cand_cpu)  # each [M*K] CPU

    warm_mk = warm.view(M, K)
    cool_mk = cool.view(M, K)
    real_mk = realism.view(M, K)
    score_mk = score_vec_cpu.view(M, K)

    out = {'tag': tag, 'M': M, 'K': K}
    # gaps on components
    ws = _gap_stats(warm_mk)
    out.update({f"warm_{k}": v for k, v in ws.items()})
    cs = _gap_stats(cool_mk)
    out.update({f"cool_{k}": v for k, v in cs.items()})
    rs = _gap_stats(real_mk)
    out.update({f"realism_{k}": v for k, v in rs.items()})
    ss = _gap_stats(score_mk)
    out.update({f"score_{k}": v for k, v in ss.items()})

    if tau_select is not None and tau_select > 0:
        probs = torch.softmax(score_mk / tau_select, dim=1)  # [M,K]
        ent = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=1)  # [M]
        out['bt_entropy_mean'] = float(ent.mean().item())
        out['bt_pmax_mean'] = float(probs.max(dim=1).values.mean().item())

    return out



@torch.no_grad()
def rgb_to_hsv_torch(x01: torch.Tensor):
    """
    x01: [B,3,H,W] in [0,1]
    returns: h,s,v each [B,H,W] in [0,1]
    """
    r, g, b = x01[:, 0], x01[:, 1], x01[:, 2]
    maxc, _ = torch.max(x01, dim=1)  # [B,H,W]
    minc, _ = torch.min(x01, dim=1)  # [B,H,W]
    v = maxc
    delta = maxc - minc + 1e-8

    s = delta / (maxc + 1e-8)

    # Hue
    h = torch.zeros_like(maxc)
    mask = delta > 1e-8

    # Where max is r
    mr = mask & (maxc == r)
    mg = mask & (maxc == g)
    mb = mask & (maxc == b)

    h[mr] = ((g[mr] - b[mr]) / delta[mr]) % 6.0
    h[mg] = ((b[mg] - r[mg]) / delta[mg]) + 2.0
    h[mb] = ((r[mb] - g[mb]) / delta[mb]) + 4.0

    h = (h / 6.0) % 1.0  # normalize to [0,1]
    return h, s, v

@torch.no_grad()
def _color_realism_terms(x_m1_1_cpu: torch.Tensor):
    """
    x_m1_1_cpu: [B,3,32,32] CPU in [-1,1]
    returns warm_term, cool_term, realism_term: all [B] on CPU
    """
    x01 = (x_m1_1_cpu.clamp(-1, 1) + 1.0) / 2.0  # [0,1]
    mu = x01.mean(dim=(2, 3))                    # [B,3]
    std = x01.std(dim=(2, 3))                    # [B,3]

    mu0 = CIFAR_MEAN.view(1, 3).to(mu.device)
    std0 = CIFAR_STD.view(1, 3).to(std.device)

    # realism: encourage global mean/std close to real data to avoid trivial monochrome hacks
    realism = -((mu - mu0).pow(2).sum(dim=1).sqrt() + (std - std0).pow(2).sum(dim=1).sqrt())

    # # warm/cool: use global channel statistics (simple but very stable)
    # warm = mu[:, 0] - mu[:, 2]  # R - B
    # cool = mu[:, 2] - mu[:, 0]  # B - R

    # warm band: red/orange/yellow (wrap-around)
    h, s, v = rgb_to_hsv_torch(x01.to(torch.device))  # x01 is [B,3,H,W] in [0,1]
    warm = hue_band_score(h, s, lo=0.92, hi=0.17, sat_pow=1.5)  # [B]
    # cool band: cyan/blue (no wrap-around)
    cool = hue_band_score(h, s, lo=0.5, hi=0.72, sat_pow=1.5)  # [B]

    return warm.cpu(), cool.cpu(), realism.cpu()

@torch.no_grad()
def quantiles_1d(x: torch.Tensor, qs=(0.1, 0.5, 0.9)) -> dict:
    """
    x: 1D tensor on CPU or GPU
    returns dict like {"p10":..., "p50":..., "p90":...}
    """
    x = x.detach().float().flatten().cpu()
    q = torch.quantile(x, torch.tensor(qs))
    out = {}
    for p, v in zip(qs, q):
        out[f"p{int(round(p*100)):02d}"] = float(v.item())
    return out

@torch.no_grad()
def reward_breakdown_stats(x_m1_1_cpu: torch.Tensor, prefix: str) -> Dict[str, float]:
    if x_m1_1_cpu.device.type != "cpu":
        x_m1_1_cpu = x_m1_1_cpu.cpu()
    warm, cool, realism = _color_realism_terms(x_m1_1_cpu)

    warm_quan = quantiles_1d(warm, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    cool_quan = quantiles_1d(cool, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    stats = {
        f'{prefix}_warm_mean': float(warm.mean().item()),
        f'{prefix}_warm_std': float(warm.std().item()),
        f'{prefix}_cool_mean': float(cool.mean().item()),
        f'{prefix}_cool_std': float(cool.std().item()),
        f'{prefix}_realism_mean': float(realism.mean().item()),
        f'{prefix}_realism_std': float(realism.std().item()),
    }
    stats.update({f'{prefix}_warm_{q}': v for q, v in warm_quan.items()})
    stats.update({f'{prefix}_cool_{q}': v for q, v in cool_quan.items()})
    return stats

    
@torch.no_grad()
def bt_pick_indices(scores_g: torch.Tensor, bt_tau: float) -> torch.Tensor:
    """
    scores_g: [M,K] CPU
    returns: best indices [M] CPU
    """
    logits = scores_g / max(bt_tau, 1e-8)
    probs = torch.softmax(logits, dim=1)
    idx = torch.multinomial(probs, num_samples=1).squeeze(1)  # [M]
    return idx
    
@torch.no_grad()
def hue_band_score(h: torch.Tensor, s: torch.Tensor, lo: float, hi: float, sat_pow: float=1.0) -> torch.Tensor:
    """
    h,s: [B,H,W] in [0,1]
    lo, hi define a hue interval. If lo <= hi: [lo, hi]. If lo > hi: wrap-around. 
    returns per-image score [B] (higher = more pixels in the band, weighted by saturation)
    """
    if lo <= hi:
        mask = (h >= lo) & (h <= hi)
    else:
        # wrap-around interval: [lo, 1] U [0, hi]
        mask = (h >= lo) | (h <= hi)
    
    w = (s.clamp(0,1) ** sat_pow)
    # average weighted occupancy
    score = (mask.float() * w).mean(dim=(1,2))  # [B]
    return score

def normalize_cifar_for_q(x_0_1: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.4914, 0.4822, 0.4465], device=x_0_1.device).view(1, 3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616], device=x_0_1.device).view(1, 3, 1, 1)
    return (x_0_1 - mean) / std

def balanced_subset_indices(dataset, size: int, num_classes: int = 10, seed: int = 0) -> List[int]:
    rng = random.Random(seed)
    per = size // num_classes
    extra = size - per * num_classes
    by_class = [[] for _ in range(num_classes)]
    for idx, y in enumerate(dataset.targets):
        by_class[y].append(idx)
    for c in range(num_classes):
        rng.shuffle(by_class[c])

    idxs = []
    for c in range(num_classes):
        take = per + (1 if c < extra else 0)
        idxs.extend(by_class[c][:take])
    rng.shuffle(idxs)
    return idxs

def counts_from_fracs(n: int, frac_real: float, frac_raw: float, frac_self_in_raw: float, frac_self_in_cur: float):
    assert 0 <= frac_real <= 1 and 0 <= frac_raw <= 1
    assert frac_real + frac_raw <= 1.0 + 1e-8
    assert 0 <= frac_self_in_raw <= 1 and 0 <= frac_self_in_cur <= 1

    n_real = int(round(n * frac_real))
    n_raw = int(round(n * frac_raw))
    n_cur = n - n_real - n_raw

    n_self_raw = int(round(n_raw * frac_self_in_raw))
    n_cross_raw = n_raw - n_self_raw

    n_self_cur = int(round(n_cur * frac_self_in_cur))
    n_cross_cur = n_cur - n_self_cur

    return {
        "n": n,
        "n_real": n_real,
        "n_raw": n_raw,
        "n_cur": n_cur,
        "n_self_raw": n_self_raw,
        "n_cross_raw": n_cross_raw,
        "n_self_cur": n_self_cur,
        "n_cross_cur": n_cross_cur,
    }


# CIFAR WideResNet
class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.drop_rate = drop_rate
        self.equal_in_out = (in_planes == out_planes)
        self.conv_shortcut = None if self.equal_in_out else nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False
        )

    def forward(self, x):
        if not self.equal_in_out:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))

        out = self.conv1(out if self.equal_in_out else x)
        out = self.relu2(self.bn2(out))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        out = self.conv2(out)

        shortcut = x if self.equal_in_out else self.conv_shortcut(x)
        return out + shortcut

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, drop_rate=0.0):
        super().__init__()
        layers = []
        for i in range(nb_layers):
            layers.append(
                block(
                    in_planes if i == 0 else out_planes,
                    out_planes,
                    stride if i == 0 else 1,
                    drop_rate,
                )
            )
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, depth=40, widen_factor=10, num_classes=10, drop_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0
        n = (depth - 4) // 6
        k = widen_factor
        n_channels = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], BasicBlock, stride=1, drop_rate=drop_rate)
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], BasicBlock, stride=2, drop_rate=drop_rate)
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], BasicBlock, stride=2, drop_rate=drop_rate)
        self.bn1 = nn.BatchNorm2d(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(n_channels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.adaptive_avg_pool2d(out, 1).view(out.size(0), -1)
        return self.fc(out)
    


# -----------------------------
# Source codes (used for per-source stats)
# -----------------------------
SRC_REAL      = 0
SRC_SELF_RAW  = 1
SRC_CROSS_RAW = 2
SRC_SELF_CUR  = 3
SRC_CROSS_CUR = 4

SRC_NAMES = ["real", "self_raw", "cross_raw", "self_cur", "cross_cur"]


def _label_hist(y_cpu: torch.Tensor, num_classes: int = 10):
    y = y_cpu.detach().cpu().long()
    return torch.bincount(y, minlength=num_classes).tolist()


def _rgb_hsv_stats_from_x_m1_1(x_m1_1_cpu: torch.Tensor):
    """
    x_m1_1_cpu: [N,3,H,W] in [-1,1] on CPU
    returns a dict of RGB + simple HSV means (all scalar floats)
    """
    x01 = (x_m1_1_cpu.clamp(-1, 1) + 1.0) / 2.0  # [0,1]
    # RGB means
    ch = x01.mean(dim=(0, 2, 3))
    r, g, b = [float(v.item()) for v in ch]

    # HSV means (vectorized)
    # Reference: standard rgb->hsv conversion
    rgb = x01  # [N,3,H,W]
    maxc, _ = rgb.max(dim=1)  # [N,H,W]
    minc, _ = rgb.min(dim=1)
    v = maxc
    deltac = (maxc - minc).clamp_min(1e-8)
    s = (deltac / maxc.clamp_min(1e-8))

    rc = (maxc - rgb[:, 0]) / deltac
    gc = (maxc - rgb[:, 1]) / deltac
    bc = (maxc - rgb[:, 2]) / deltac

    # hue in [0,1)
    h = torch.zeros_like(maxc)
    mask_r = (rgb[:, 0] == maxc)
    mask_g = (rgb[:, 1] == maxc)
    mask_b = (rgb[:, 2] == maxc)

    h[mask_r] = (bc - gc)[mask_r]
    h[mask_g] = 2.0 + (rc - bc)[mask_g]
    h[mask_b] = 4.0 + (gc - rc)[mask_b]
    h = (h / 6.0) % 1.0

    # If grayscale (max==min), define s=0 and h=0 already ok due to deltac clamp
    mean_h = float(h.mean().item())
    mean_s = float(s.mean().item())
    mean_v = float(v.mean().item())

    return {
        "mean_r": r,
        "mean_g": g,
        "mean_b": b,
        "mean_r_minus_b": r - b,
        "mean_b_minus_r": b - r,
        "mean_rgb": float(ch.mean().item()),
        "mean_h": mean_h,
        "mean_s": mean_s,
        "mean_v": mean_v,
    }


def _per_source_stats(x_m1_1_cpu: torch.Tensor,
                      y_cpu: torch.Tensor,
                      src_cpu: torch.Tensor,
                      num_classes: int = 10):
    """
    Returns dict keyed by SRC_NAMES[*] with n/label_hist/rgb_hsv stats.
    """
    out = {}
    src_cpu = src_cpu.detach().cpu().long()
    for s in torch.unique(src_cpu).tolist():
        mask = (src_cpu == int(s))
        if mask.sum().item() == 0:
            continue
        name = SRC_NAMES[int(s)] if 0 <= int(s) < len(SRC_NAMES) else str(int(s))
        out[name] = {
            "n": int(mask.sum().item()),
            "label_hist": _label_hist(y_cpu[mask], num_classes=num_classes),
            "rgb_hsv": _rgb_hsv_stats_from_x_m1_1(x_m1_1_cpu[mask]),
        }
    return out
