import os
import json
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict
import argparse
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image
from tqdm import tqdm
import torchvision.transforms.functional as TF

from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
from scripts.utils import *
from scripts.utils import _label_hist, _rgb_hsv_stats_from_x_m1_1, _per_source_stats, _color_realism_terms, summarize_candidate_diffs


# Pools
class TensorPool:
    def __init__(self, images_m1_1_cpu: torch.Tensor, labels_cpu: torch.Tensor, name: str):
        assert images_m1_1_cpu.device.type == "cpu"
        assert labels_cpu.device.type == "cpu"
        self.images = images_m1_1_cpu
        self.labels = labels_cpu.long()
        self.name = name

    @property
    def n(self) -> int:
        return int(self.labels.numel())

    def sample_images(self, m: int, gen: torch.Generator) -> torch.Tensor:
        idx = torch.randint(0, self.n, (m,), generator=gen)
        return self.images[idx]

    def sample_labels(self, m: int, gen: torch.Generator) -> torch.Tensor:
        idx = torch.randint(0, self.n, (m,), generator=gen)
        return self.labels[idx]

    def sample_pairs(self, m: int, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, self.n, (m,), generator=gen)
        return self.images[idx], self.labels[idx]


# Diffusion sampling (DDIM) with generator
@torch.no_grad()
def sample_ddim_images(
    unet: UNet2DModel,
    scheduler: DDIMScheduler,
    labels_dev: torch.Tensor,
    num_steps: int,
    device: torch.device,
    gen: torch.Generator,
    batch: int = 256,
) -> torch.Tensor:
    """
    labels_dev: [N] on device
    returns: images [-1,1], [N,3,32,32] on CPU
    """
    unet.eval()
    N = labels_dev.size(0)
    scheduler.set_timesteps(num_steps, device=device)
    outs = []
    for start in range(0, N, batch):
        end = min(N, start + batch)
        y = labels_dev[start:end]
        x = torch.randn(y.size(0), 3, 32, 32, device=device, generator=gen)
        for t in scheduler.timesteps:
            eps = unet(x, t, class_labels=y).sample
            x = scheduler.step(eps, t, x).prev_sample
        outs.append(x.detach().cpu())
    return torch.cat(outs, dim=0)


# Args
@dataclass
class Args:
    out_dir: str = "runs/cifar_pq_v3_repro"
    seed: int = 123
    device: str = "cuda"
    fp16: bool = True

    # fixed real split sizes
    real_train_size: int = 10000
    real_test_size: int = 2000
    balanced_real_subset: bool = True

    # per-iter trainset sizes (fixed n)
    trainset_size_p: int = 8000
    trainset_size_q: int = 8000

    # composition fractions for p
    frac_real_p: float = 0.5
    frac_raw_p: float = 0.3
    frac_self_in_raw_p: float = 0.7
    frac_self_in_cur_p: float = 0.7

    # composition fractions for q
    frac_real_q: float = 0.5
    frac_raw_q: float = 0.3
    frac_self_in_raw_q: float = 0.7
    frac_self_in_cur_q: float = 0.7

    # curated K candidates
    k_candidates: int = 4

    # diffusion sampling
    ddim_steps_train: int = 30
    ddim_steps_eval: int = 50
    gen_batch: int = 256

    # loop/training
    iters: int = 55
    train_steps_p: int = 800
    train_steps_q: int = 800
    bs_p: int = 64
    bs_q: int = 128
    lr_p: float = 2e-4
    lr_q: float = 1e-4
    weight_decay: float = 0.01

    # reward
    # p-image reward for curation: rp = lamA*A + lamH*H(q_probs)
    lamA: float = 1.0
    lamH: float = 0.0
    # q reward terms:  rq = w_clip * s_vec + w_logq * log q_prob
    w_clip: float = 1.0
    w_logq: float = 1.0

    # q-image reward for curation: rq_img = <q_probs, s_vec> - tau*H(q_probs)
    tau: float = 0.2

    clip_batch: int = 64

    # eval
    eval_real_samples: int = 512

    # train-time augmentation
    aug_pq_flip: int = 0           # 1=enable p horizontal flip augmentation
    aug_pq_flip_pq: float = 0.5      # flip prob for p

    init_p_path: str = None
    init_q_path: str = None

    color_alpha: float = 1.0
    realism_beta: float = 0.5

    bt_tau: float = 1.0

    lr_decay_start: int = 40      # decay learning rate after this epoch (inclusive)
    lr_decay_gamma: float = 0.1   # learning rate decay factor (new_lr = old_lr * gamma)
    lr_decay_every: int = 1       # decay learning rate every this many epochs (1=each epoch; 2=every 2 epochs, etc.)
    wd_after_decay: float = 0.0   # weight decay during decay phase (0=disable)

def main(args):
    def set_opt_hparams(opt: torch.optim.Optimizer, lr: float, weight_decay: float):
        for param_group in opt.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = args.fp16 and (device.type == "cuda")

    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "images"))
    ensure_dir(os.path.join(args.out_dir, "ckpt"))

    loss_csv_path = os.path.join(args.out_dir, "loss_by_epoch.csv")
    loss_csv_f = open(loss_csv_path, "a", newline='')
    loss_writer = csv.DictWriter(
        loss_csv_f,
        fieldnames=["epoch", "lr_p", "lr_q", "p_last_loss", "q_last_loss", "p_mean_loss", "q_mean_loss"]
    )

    # write header once
    if loss_csv_f.tell() == 0:
        loss_writer.writeheader()
        loss_csv_f.flush()

    # models
    unet_p = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        # layers_per_block=2,
        layers_per_block=1,
        # block_out_channels=(128, 256, 256, 512),
        block_out_channels=(96, 192, 192, 384),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        num_class_embeds=10,
    ).to(device)

    unet_q = UNet2DModel(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        # layers_per_block=2,
        layers_per_block=1,
        # block_out_channels=(128, 256, 256, 512),
        block_out_channels=(96, 192, 192, 384),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
        num_class_embeds=10,
    ).to(device)

    if args.init_p_path is not None:
        print(f"Loading p init from {args.init_p_path}")
        ckpt_p = torch.load(args.init_p_path, map_location="cpu")
        unet_p.load_state_dict(ckpt_p['unet'] if 'unet' in ckpt_p else ckpt_p, strict=True)

    if args.init_q_path is not None:
        print(f"Loading q init from {args.init_q_path}")
        ckpt_q = torch.load(args.init_q_path, map_location="cpu")
        unet_q.load_state_dict(ckpt_q['unet'] if 'unet' in ckpt_q else ckpt_q, strict=True)

    ddpm = DDPMScheduler(num_train_timesteps=1000)
    ddim = DDIMScheduler.from_config(ddpm.config)

    start_epoch = 0

    opt_q = torch.optim.AdamW(unet_q.parameters(), lr=args.lr_q, weight_decay=args.weight_decay)
    opt_p = torch.optim.AdamW(unet_p.parameters(), lr=args.lr_p, weight_decay=args.weight_decay)

    # AMP scalers (per your requirement)
    scaler_q = torch.amp.GradScaler(device_type, enabled=amp_enabled)
    scaler_p = torch.amp.GradScaler(device_type, enabled=amp_enabled)


    # fixed real split
    transform = T.Compose([T.ToTensor()])
    cifar_train_full = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    cifar_test_full = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

    if args.balanced_real_subset:
        tr_idx = balanced_subset_indices(cifar_train_full, args.real_train_size, seed=args.seed)
        te_idx = balanced_subset_indices(cifar_test_full, args.real_test_size, seed=args.seed + 1)
    else:
        tr_idx = list(range(args.real_train_size))
        te_idx = list(range(args.real_test_size))

    def materialize(dataset, indices: List[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for i in indices:
            x, y = dataset[i]  # x [0,1]
            xs.append(x)
            ys.append(y)
        x01 = torch.stack(xs, dim=0)
        y = torch.tensor(ys, dtype=torch.long)
        xm1 = x01 * 2.0 - 1.0
        return xm1.cpu(), y.cpu()

    real_train_x, real_train_y = materialize(cifar_train_full, tr_idx)
    real_test_x, real_test_y = materialize(cifar_test_full, te_idx)

    real_train_pool = TensorPool(real_train_x, real_train_y, "real_train")
    real_test_pool = TensorPool(real_test_x, real_test_y, "real_test")

    m_eval = min(args.eval_real_samples, real_test_pool.n)
    gen_eval = torch.Generator(device="cpu").manual_seed(args.seed + 2026)
    perm = torch.randperm(real_test_pool.n, generator=gen_eval)
    eval_idx = perm[:m_eval]

    eval_x_cpu = real_test_pool.images[eval_idx].contiguous()
    eval_y_cpu = real_test_pool.labels[eval_idx].contiguous()


    # train-time augmentation functions, tensor-only
    def _rand_hflip(x: torch.Tensor, p: float) -> torch.Tensor:
        # x: [B,C,H,W]
        if p <= 0:
            return x
        B = x.size(0)
        mask = (torch.rand(B, device=x.device) < p)
        if mask.any():
            x = x.clone()  # avoid in-place side-effects
            x[mask] = torch.flip(x[mask], dims=[3])  # flip width
        return x

    def aug_pq_train(x_m1_1: torch.Tensor) -> torch.Tensor:
        # p uses flip only (safe for diffusion); x in [-1,1]
        x_m1_1 = _rand_hflip(x_m1_1, args.aug_pq_flip_pq)
        return x_m1_1   
    
    @torch.no_grad()
    def p_reward_warm(x_m1_1_cpu: torch.Tensor) -> torch.Tensor:
        warm, _, realism = _color_realism_terms(x_m1_1_cpu)
        return args.color_alpha * warm + args.realism_beta * realism

    @torch.no_grad()
    def q_reward_cool(x_m1_1_cpu: torch.Tensor) -> torch.Tensor:
        _, cool, realism = _color_realism_terms(x_m1_1_cpu)
        return args.color_alpha * cool + args.realism_beta * realism

    # training steps
    def train_q_on_pool(train_pool: TensorPool, steps: int, gen_cpu_train: torch.Generator, gen_dev_train: torch.Generator):
        unet_q.train()
        last_loss = 0.0
        sum_loss = 0.0

        pbar = tqdm(range(steps), desc="train q(diffusion)", leave=False)
        for _ in pbar:
            x_cpu, y_cpu = train_pool.sample_pairs(args.bs_q, gen=gen_cpu_train)
            x_dev = x_cpu.to(device)
            y_dev = y_cpu.to(device)

            if args.aug_pq_flip == 1:
                x_dev = aug_pq_train(x_dev)

            bs = x_dev.size(0)
            t = torch.randint(0, ddpm.config.num_train_timesteps, (bs,), device=device, generator=gen_dev_train).long()
            noise = torch.randn(bs, 3, 32, 32, device=device, generator=gen_dev_train)
            x_noisy = ddpm.add_noise(x_dev, noise, t)

            opt_q.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type, enabled=amp_enabled):
                eps_pred = unet_q(x_noisy, t, class_labels=y_dev).sample
                loss = F.mse_loss(eps_pred, noise)

            scaler_q.scale(loss).backward()
            scaler_q.step(opt_q)
            scaler_q.update()
            last_loss = float(loss.detach().cpu())
            sum_loss += last_loss
            pbar.set_postfix({"loss": last_loss})
        
        mean_loss = sum_loss / steps
        return last_loss, mean_loss

    def train_p_on_pool(train_pool: TensorPool, steps: int, gen_cpu_train: torch.Generator, gen_dev_train: torch.Generator):
        unet_p.train()
        last_loss = 0.0
        sum_loss = 0.0

        pbar = tqdm(range(steps), desc="train p", leave=False)
        for _ in pbar:
            x_cpu, y_cpu = train_pool.sample_pairs(args.bs_p, gen=gen_cpu_train)
            x_dev = x_cpu.to(device)
            y_dev = y_cpu.to(device)

            if args.aug_pq_flip == 1:
                x_dev = aug_pq_train(x_dev)

            bs = x_dev.size(0)
            t = torch.randint(0, ddpm.config.num_train_timesteps, (bs,), device=device, generator=gen_dev_train).long()
            noise = torch.randn(bs, 3, 32, 32, device=device, generator=gen_dev_train)
            x_noisy = ddpm.add_noise(x_dev, noise, t)

            opt_p.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type, enabled=amp_enabled):
                eps_pred = unet_p(x_noisy, t, class_labels=y_dev).sample
                loss = F.mse_loss(eps_pred, noise)

            scaler_p.scale(loss).backward()
            scaler_p.step(opt_p)
            scaler_p.update()

            last_loss = float(loss.detach().cpu())
            sum_loss += last_loss
            pbar.set_postfix({"loss": last_loss})
        
        mean_loss = sum_loss / steps
        return last_loss, mean_loss
    
    def _merge_cand_stats(stats: dict, cand_stats: dict, prefix: str):
        for k, v in cand_stats.items():
            if k == 'tag' or k == 'M' or k == 'K':
                continue
            stats[f"{prefix}_{k}"] = v

    # build per-iter trainsets
    def build_trainset_for_p(
        prev_p: TensorPool,
        prev_q: TensorPool,
        gen_cpu_build: torch.Generator,
        gen_dev_sample: torch.Generator,
    ) -> Tuple[TensorPool, Dict[str, int]]:
        cnt = counts_from_fracs(
            args.trainset_size_p,
            args.frac_real_p,
            args.frac_raw_p,
            args.frac_self_in_raw_p,
            args.frac_self_in_cur_p,
        )
        xs, ys = [], []
        srcs = [] # 0=real, 1=self_raw, 2=cross_raw, 3=self_cur, 4=cross_cur; used for analysis but not training
        stats_out = {}

        # real
        if cnt["n_real"] > 0:
            x_real, y_real = real_train_pool.sample_pairs(cnt["n_real"], gen=gen_cpu_build)
            xs.append(x_real); ys.append(y_real)
            srcs.append(torch.full((y_real.size(0),), SRC_REAL, dtype=torch.long))  # real label -1

        # raw self (p): labels from prev_p -> 1 image each
        if cnt["n_self_raw"] > 0:
            y = prev_p.sample_labels(cnt["n_self_raw"], gen=gen_cpu_build).to(device)
            x = sample_ddim_images(unet_p, ddim, y, args.ddim_steps_train, device, gen=gen_dev_sample, batch=args.gen_batch)
            xs.append(x); ys.append(y.detach().cpu())
            srcs.append(torch.full((y.size(0),), SRC_SELF_RAW, dtype=torch.long))

        # raw cross (q -> p): images from prev_q -> q predicts 1 label; swap to (label,image) by storing (image,label)
        if cnt["n_cross_raw"] > 0:
            y = prev_q.sample_labels(cnt["n_cross_raw"], gen=gen_cpu_build).to(device)
            x = sample_ddim_images(unet_q, ddim, y, args.ddim_steps_train, device, gen=gen_dev_sample, batch=args.gen_batch)
            xs.append(x); ys.append(y.detach().cpu())
            srcs.append(torch.full((y.size(0),), SRC_CROSS_RAW, dtype=torch.long))

        # curated self (p): per label generate K images, choose by p-reward
        if cnt["n_self_cur"] > 0:
            M, K = cnt["n_self_cur"], args.k_candidates
            y_base = prev_p.sample_labels(M, gen=gen_cpu_build).to(device)  # [M]
            y_rep = y_base.repeat_interleave(K)                             # [M*K]
            x_cand = sample_ddim_images(unet_p, ddim, y_rep, args.ddim_steps_train, device, gen=gen_dev_sample, batch=args.gen_batch)  # CPU [M*K]
            scores = p_reward_warm(x_cand)  # CPU [M*K]
            
            cand_stats = summarize_candidate_diffs(
                x_cand_cpu=x_cand, M=M, K=K, score_vec_cpu=scores, tag="p_self_cur", tau_select=args.bt_tau
            )
            _merge_cand_stats(stats_out, cand_stats, prefix="p_self_cur")

            scores_g = scores.view(M, K)
            # best = scores_g.argmax(dim=1)
            best = bt_pick_indices(scores_g, args.bt_tau)
            x_cand_g = x_cand.view(M, K, 3, 32, 32)
            x_best = x_cand_g[torch.arange(M), best]
            xs.append(x_best); ys.append(y_base.detach().cpu())
            srcs.append(torch.full((y_base.size(0),), SRC_SELF_CUR, dtype=torch.long))

        # curated cross (q -> p): images from prev_q, sample K labels and choose by CLIP taxonomy via q_choose
        if cnt["n_cross_cur"] > 0:
            M, K = cnt["n_cross_cur"], args.k_candidates
            y_base = prev_q.sample_labels(M, gen=gen_cpu_build).to(device)  # [M]
            y_rep = y_base.repeat_interleave(K)                             # [M*K]
            x_cand = sample_ddim_images(unet_q, ddim, y_rep, args.ddim_steps_train, device, gen=gen_dev_sample, batch=args.gen_batch)  # CPU [M*K]
            scores = q_reward_cool(x_cand)  # CPU [M*K]

            cand_stats = summarize_candidate_diffs(
                x_cand_cpu=x_cand, M=M, K=K, score_vec_cpu=scores, tag="p_cross_cur", tau_select=args.bt_tau
            )
            _merge_cand_stats(stats_out, cand_stats, prefix="p_cross_cur")

            scores_g = scores.view(M, K)
            # best = scores_g.argmax(dim=1)
            best = bt_pick_indices(scores_g, args.bt_tau)
            x_cand_g = x_cand.view(M, K, 3, 32, 32)
            x_best = x_cand_g[torch.arange(M), best]
            xs.append(x_best); ys.append(y_base.detach().cpu())
            srcs.append(torch.full((y_base.size(0),), SRC_CROSS_CUR, dtype=torch.long))

        x_all = torch.cat(xs, dim=0) if xs else torch.empty(0, 3, 32, 32)
        y_all = torch.cat(ys, dim=0) if ys else torch.empty(0, dtype=torch.long)

        # trim/pad to exact size
        if x_all.size(0) > cnt["n"]:
            x_all = x_all[:cnt["n"]]; y_all = y_all[:cnt["n"]]
        elif x_all.size(0) < cnt["n"]:
            need = cnt["n"] - x_all.size(0)
            x_pad, y_pad = real_train_pool.sample_pairs(need, gen=gen_cpu_build)
            x_all = torch.cat([x_all, x_pad], dim=0)
            y_all = torch.cat([y_all, y_pad], dim=0)

        src_all = torch.cat(srcs, dim=0) if srcs else torch.empty(0, dtype=torch.long)

        stats = dict(cnt)
        stats["built"] = int(x_all.size(0))
        stats.update({
            'label_hist': _label_hist(y_all, num_classes=10),
            'rgb_hsv': _rgb_hsv_stats_from_x_m1_1(x_all),
            'by_source': _per_source_stats(x_all, y_all, src_all),
        })
        stats.update(stats_out)
        return TensorPool(x_all.cpu(), y_all.cpu(), "train_p"), stats

    def build_trainset_for_q(
        prev_p: TensorPool,
        prev_q: TensorPool,
        gen_cpu_build: torch.Generator,
        gen_dev_sample: torch.Generator,
    ) -> Tuple[TensorPool, Dict[str, int]]:
        cnt = counts_from_fracs(
            args.trainset_size_q,
            args.frac_real_q,
            args.frac_raw_q,
            args.frac_self_in_raw_q,
            args.frac_self_in_cur_q,
        )
        xs, ys = [], []
        srcs = []  # 0=real, 1=self_raw, 2=cross_raw, 3=self_cur, 4=cross_cur; used for analysis but not training
        stats_out = {}

        # real
        if cnt["n_real"] > 0:
            x_real, y_real = real_train_pool.sample_pairs(cnt["n_real"], gen=gen_cpu_build)
            xs.append(x_real); ys.append(y_real)
            srcs.append(torch.full((y_real.size(0),), SRC_REAL, dtype=torch.long))  # real label -1

        # raw self (q): images from prev_q -> q predicts 1 label
        if cnt["n_self_raw"] > 0:
            y = prev_q.sample_labels(cnt["n_self_raw"], gen=gen_cpu_build).to(device)
            x = sample_ddim_images(unet_q, ddim, y, args.ddim_steps_train, device, gen=gen_dev_sample, batch=args.gen_batch)
            xs.append(x); ys.append(y.detach().cpu())
            srcs.append(torch.full((y.size(0),), SRC_SELF_RAW, dtype=torch.long))

        # raw cross (p -> q): labels from prev_p -> 1 image each
        if cnt["n_cross_raw"] > 0:
            y = prev_p.sample_labels(cnt["n_cross_raw"], gen=gen_cpu_build).to(device)
            x = sample_ddim_images(unet_p, ddim, y, args.ddim_steps_train, device, gen=gen_dev_sample, batch=args.gen_batch)
            xs.append(x); ys.append(y.detach().cpu())
            srcs.append(torch.full((y.size(0),), SRC_CROSS_RAW, dtype=torch.long))

        # curated self (q): images from prev_q -> sample K labels and choose by CLIP taxonomy
        if cnt["n_self_cur"] > 0:
            M, K = cnt["n_self_cur"], args.k_candidates
            y_base = prev_q.sample_labels(M, gen=gen_cpu_build).to(device)  # [M]
            y_rep = y_base.repeat_interleave(K)
            x_cand = sample_ddim_images(unet_q, ddim, y_rep, args.ddim_steps_train, device, gen=gen_dev_sample, batch=args.gen_batch)  # CPU [M*K]
            scores = q_reward_cool(x_cand)  # CPU [M*K]

            cand_stats = summarize_candidate_diffs(
                x_cand_cpu=x_cand, M=M, K=K, score_vec_cpu=scores, tag="q_self_cur", tau_select=args.bt_tau
            )
            _merge_cand_stats(stats_out, cand_stats, prefix="q_self_cur")

            scores_g = scores.view(M, K)
            # best = scores_g.argmax(dim=1)
            best = bt_pick_indices(scores_g, args.bt_tau)
            x_cand_g = x_cand.view(M, K, 3, 32, 32)
            x_best = x_cand_g[torch.arange(M), best]
            xs.append(x_best); ys.append(y_base.detach().cpu())
            srcs.append(torch.full((y_base.size(0),), SRC_SELF_CUR, dtype=torch.long))

        # curated cross (p -> q): per label generate K images, choose by q-image reward
        if cnt["n_cross_cur"] > 0:
            M, K = cnt["n_cross_cur"], args.k_candidates
            y_base = prev_p.sample_labels(M, gen=gen_cpu_build).to(device)  # [M]
            y_rep = y_base.repeat_interleave(K)
            x_cand = sample_ddim_images(unet_p, ddim, y_rep, args.ddim_steps_train, device, gen=gen_dev_sample, batch=args.gen_batch)  # CPU [M*K]
            scores = p_reward_warm(x_cand)  # CPU [M*K]

            cand_stats = summarize_candidate_diffs(
                x_cand_cpu=x_cand, M=M, K=K, score_vec_cpu=scores, tag="q_cross_cur", tau_select=args.bt_tau
            )
            _merge_cand_stats(stats_out, cand_stats, prefix="q_cross_cur")

            scores_g = scores.view(M, K)
            # best = scores_g.argmax(dim=1)
            best = bt_pick_indices(scores_g, args.bt_tau)
            x_cand_g = x_cand.view(M, K, 3, 32, 32)
            x_best = x_cand_g[torch.arange(M), best]
            xs.append(x_best); ys.append(y_base.detach().cpu())
            srcs.append(torch.full((y_base.size(0),), SRC_CROSS_CUR, dtype=torch.long))

        x_all = torch.cat(xs, dim=0) if xs else torch.empty(0, 3, 32, 32)
        y_all = torch.cat(ys, dim=0) if ys else torch.empty(0, dtype=torch.long)
        src_all = torch.cat(srcs, dim=0) if srcs else torch.empty(0, dtype=torch.long)

        # trim/pad
        if x_all.size(0) > cnt["n"]:
            x_all = x_all[:cnt["n"]]; y_all = y_all[:cnt["n"]]
        elif x_all.size(0) < cnt["n"]:
            need = cnt["n"] - x_all.size(0)
            x_pad, y_pad = real_train_pool.sample_pairs(need, gen=gen_cpu_build)
            x_all = torch.cat([x_all, x_pad], dim=0)
            y_all = torch.cat([y_all, y_pad], dim=0)

        stats = dict(cnt)
        stats["built"] = int(x_all.size(0))
        stats.update(stats_out)
        stats.update({
            'label_hist': _label_hist(y_all, num_classes=10),
            'rgb_hsv': _rgb_hsv_stats_from_x_m1_1(x_all),
            'by_source': _per_source_stats(x_all, y_all, src_all),
        })
        return TensorPool(x_all.cpu(), y_all.cpu(), "train_q"), stats

    # init prev trainsets (iter 0)
    gen_cpu_init = torch.Generator(device="cpu").manual_seed(args.seed + 777)
    x0p, y0p = real_train_pool.sample_pairs(args.trainset_size_p, gen=gen_cpu_init)
    x0q, y0q = real_train_pool.sample_pairs(args.trainset_size_q, gen=gen_cpu_init)
    prev_train_p = TensorPool(x0p, y0p, name="prev_train_p_init")
    prev_train_q = TensorPool(x0q, y0q, name="prev_train_q_init")
    print('prev_train_q.n = ', prev_train_q.n)
    
    @torch.no_grad()
    def eval_gen_and_save(it: int) -> Dict[str, float]:
        unet_q.eval()
        unet_p.eval()

        y = eval_y_cpu.to(device)
        # Fixed eval noise seed each iter => comparable across iters (only model params change)
        if device.type == "cuda":
            gen_dev_eval = torch.Generator(device="cuda").manual_seed(args.seed + 888888)
        else:
            gen_dev_eval = torch.Generator(device="cpu").manual_seed(args.seed + 888888)
        
        x_p = sample_ddim_images(unet_p, ddim, y, args.ddim_steps_eval, device, batch=args.gen_batch, gen=gen_dev_eval)  # CPU [-1,1]
        x_q = sample_ddim_images(unet_q, ddim, y, args.ddim_steps_eval, device, batch=args.gen_batch, gen=gen_dev_eval)  # CPU [-1,1]

        rp_p = p_reward_warm(x_p)  # CPU [m]
        rp_q = p_reward_warm(x_q)  # CPU [m]
        rq_p = q_reward_cool(x_p)  # CPU [m]
        rq_q = q_reward_cool(x_q)  # CPU [m]

        rp_p_quan = quantiles_1d(rp_p, qs=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))
        rp_q_quan = quantiles_1d(rp_q, qs=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))
        rq_p_quan = quantiles_1d(rq_p, qs=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))
        rq_q_quan = quantiles_1d(rq_q, qs=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0))

        x_p_cpu_01 = (x_p.clamp(-1, 1) + 1.0) / 2.0
        x_q_cpu_01 = (x_q.clamp(-1, 1) + 1.0) / 2.0
        # save grids for inspection
        grid_path_p = os.path.join(args.out_dir, "images", f"eval_gen_p_from_true_y_iter_{it:03d}.png")
        grid_path_q = os.path.join(args.out_dir, "images", f"eval_gen_q_from_true_y_iter_{it:03d}.png")
        save_n = min(256, x_p.size(0))
        save_image(x_p_cpu_01[:save_n], grid_path_p, nrow=16)
        save_image(x_q_cpu_01[:save_n], grid_path_q, nrow=16)

        out = {
                "p_reward_on_pgen(p reward)": float(rp_p.mean().item()),
                "p_reward_on_qgen": float(rp_q.mean().item()),
                "q_reward_on_pgen": float(rq_p.mean().item()),
                "q_reward_on_qgen(q reward)": float(rq_q.mean().item())
            }
        out.update({f"p_reward_on_pgen_{k}": v for k, v in rp_p_quan.items()})
        out.update({f"p_reward_on_qgen_{k}": v for k, v in rp_q_quan.items()})
        out.update({f"q_reward_on_pgen_{k}": v for k, v in rq_p_quan.items()})
        out.update({f"q_reward_on_qgen_{k}": v for k, v in rq_q_quan.items()})
        out.update(reward_breakdown_stats(x_p, "p"))
        out.update(reward_breakdown_stats(x_q, "q"))    
        out.update({"eval_rgb_hsv_pgen": _rgb_hsv_stats_from_x_m1_1(x_p)})
        out.update({"eval_rgb_hsv_qgen": _rgb_hsv_stats_from_x_m1_1(x_q)})

        return out
    
    # Iterative loop
    lr_p, lr_q = args.lr_p, args.lr_q
    for it in range(start_epoch, start_epoch + args.iters):
        # Per-iteration deterministic generators (different each iter)
        gen_cpu_build = torch.Generator(device="cpu").manual_seed(args.seed + it * 10000 + 1)
        gen_cpu_train = torch.Generator(device="cpu").manual_seed(args.seed + it * 10000 + 2)

        if device.type == "cuda":
            gen_dev_sample = torch.Generator(device="cuda").manual_seed(args.seed + it * 10000 + 3)
            gen_dev_train_p = torch.Generator(device="cuda").manual_seed(args.seed + it * 10000 + 4)
        else:
            gen_dev_sample = torch.Generator(device="cpu").manual_seed(args.seed + it * 10000 + 3)
            gen_dev_train_p = torch.Generator(device="cpu").manual_seed(args.seed + it * 10000 + 4)

        # 1) build fixed-size trainsets using prev trainsets as input distributions
        train_q, stat_q = build_trainset_for_q(prev_train_p, prev_train_q, gen_cpu_build, gen_dev_sample)
        train_p, stat_p = build_trainset_for_p(prev_train_p, prev_train_q, gen_cpu_build, gen_dev_sample)

        if it >= args.lr_decay_start: 
            k = (it - args.lr_decay_start) // max(args.lr_decay_every, 1)

            if it == args.lr_decay_start:
                lr_p = opt_p.param_groups[0]['lr']
                lr_q = opt_q.param_groups[0]['lr']

            new_lr_p = lr_p * (args.lr_decay_gamma ** k)
            new_lr_q = lr_q * (args.lr_decay_gamma ** k)

            set_opt_hparams(opt_p, lr=new_lr_p, weight_decay=args.wd_after_decay)
            set_opt_hparams(opt_q, lr=new_lr_q, weight_decay=args.wd_after_decay)

        # 2) train 
        p_last_loss, p_mean_loss = train_p_on_pool(train_p, args.train_steps_p, gen_cpu_train, gen_dev_train_p)
        q_last_loss, q_mean_loss = train_q_on_pool(train_q, args.train_steps_q, gen_cpu_train, gen_dev_train_p)

        loss_writer.writerow({
            "epoch": it, 
            "lr_p": opt_p.param_groups[0]['lr'],
            "lr_q": opt_q.param_groups[0]['lr'],
            "p_last_loss": p_last_loss, 
            "p_mean_loss": p_mean_loss,
            "q_last_loss": q_last_loss,
            "q_mean_loss": q_mean_loss, 
        })
        loss_csv_f.flush()
        
        # 3) eval
        eval_results = eval_gen_and_save(it)

        # if it == 0 or it % 3 == 0:
        log = {
            "iter": it,
            "p_stats": stat_p,
            "q_stats": stat_q,
            **eval_results,
        }
        print(json.dumps(log, ensure_ascii=False))
        # save log
        out_log_path = os.path.join(args.out_dir, "log.jsonl")
        with open(out_log_path, "a") as f:
            f.write(json.dumps(log, ensure_ascii=False) + "\n")

        # 4) update prev trainsets
        prev_train_p = train_p
        prev_train_q = train_q

        if it < args.lr_decay_start:
            if it % 5 == 0 or it == args.lr_decay_start - 1:
                # checkpoints
                ckpt_dir = os.path.join(args.out_dir, "ckpt")
                torch.save({"iter": it,
                            "unet": unet_p.state_dict(), 
                            "opt_p": opt_p.state_dict(),
                            "scaler_p": scaler_p.state_dict(),
                            }, os.path.join(ckpt_dir, f"p_unet_iter_{it:03d}.pt"))
                torch.save({"iter": it,
                            "q": unet_q.state_dict(), 
                            "opt_q": opt_q.state_dict(),
                            "scaler_q": scaler_q.state_dict(),
                            }, os.path.join(ckpt_dir, f"q_unet_iter_{it:03d}.pt"))
        else:
            if it == args.iters - 1 or it % 2 == 0 or it == 49:
                # checkpoints
                ckpt_dir = os.path.join(args.out_dir, "ckpt")
                torch.save({"iter": it,
                            "unet": unet_p.state_dict(), 
                            "opt_p": opt_p.state_dict(),
                            "scaler_p": scaler_p.state_dict(),
                            }, os.path.join(ckpt_dir, f"p_unet_iter_{it:03d}.pt"))
                torch.save({"iter": it,
                            "q": unet_q.state_dict(), 
                            "opt_q": opt_q.state_dict(),
                            "scaler_q": scaler_q.state_dict(),
                            }, os.path.join(ckpt_dir, f"q_unet_iter_{it:03d}.pt"))
    
    loss_csv_f.close()


if __name__ == "__main__":
    def build_parser() -> argparse.ArgumentParser:
        p = argparse.ArgumentParser()
        p.add_argument("--out_dir", type=str, default="runs/cifar_pq_v2")
        p.add_argument("--seed", type=int, default=123)
        p.add_argument("--device", type=str, default="cuda")
        p.add_argument("--fp16", action="store_true")
        p.add_argument("--real_train_size", type=int, default=10000)
        p.add_argument("--real_test_size", type=int, default=1000)
        p.add_argument("--balanced_real_subset", action="store_true")

        p.add_argument("--trainset_size_p", type=int, default=2000)
        p.add_argument("--trainset_size_q", type=int, default=2000)

        p.add_argument("--frac_real_p", type=float, default=0.5)
        p.add_argument("--frac_raw_p", type=float, default=0.3)
        p.add_argument("--frac_self_in_raw_p", type=float, default=0.7)
        p.add_argument("--frac_self_in_cur_p", type=float, default=0.7)
        p.add_argument("--frac_real_q", type=float, default=0.5)
        p.add_argument("--frac_raw_q", type=float, default=0.3)
        p.add_argument("--frac_self_in_raw_q", type=float, default=0.7)
        p.add_argument("--frac_self_in_cur_q", type=float, default=0.7)

        p.add_argument("--k_candidates", type=int, default=6)

        p.add_argument("--ddim_steps_train", type=int, default=30)
        p.add_argument("--ddim_steps_eval", type=int, default=50)
        p.add_argument("--gen_batch", type=int, default=256)
        p.add_argument("--iters", type=int, default=55)
        p.add_argument("--train_steps_p", type=int, default=800)
        p.add_argument("--train_steps_q", type=int, default=800)
        p.add_argument("--bs_p", type=int, default=256)
        p.add_argument("--bs_q", type=int, default=256)
        p.add_argument("--lr_p", type=float, default=2e-4)
        p.add_argument("--lr_q", type=float, default=2e-4)
        p.add_argument("--weight_decay", type=float, default=5e-4)

        p.add_argument("--lamA", type=float, default=1.0)
        p.add_argument("--lamH", type=float, default=0.0)
        p.add_argument("--tau", type=float, default=0.2)

        p.add_argument("--clip_batch", type=int, default=64)
        p.add_argument("--eval_real_samples", type=int, default=512)
        p.add_argument("--w_clip", type=float, default=1.0)
        p.add_argument("--w_logq", type=float, default=1.0)
        
        p.add_argument("--aug_pq_flip", type=int, default=0)
        p.add_argument("--aug_pq_flip_pq", type=float, default=0.5)
        p.add_argument("--init_p_path", type=str, default=None)
        p.add_argument("--init_q_path", type=str, default=None)
        p.add_argument("--color_alpha", type=float, default=1.0)
        p.add_argument("--realism_beta", type=float, default=0.5)

        p.add_argument("--bt_tau", type=float, default=1.0)

        p.add_argument("--lr_decay_start", type=int, default=999999)
        p.add_argument("--wd_after_decay", type=float, default=0.0)
        p.add_argument("--lr_decay_every", type=int, default=1)  
        p.add_argument("--lr_decay_gamma", type=float, default=0.5)

        return p

    def parse_args_to_dataclass(parser: argparse.ArgumentParser) -> Args:
        parsed = parser.parse_args()
        args = Args(
            **vars(parsed)
        )
        return args
        
    args = parse_args_to_dataclass(build_parser())
    main(args)
