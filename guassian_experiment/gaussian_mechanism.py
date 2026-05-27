"""
High-dimensional exact mechanism experiment for the coupled system.
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class Config:
    output_dir: str = "output/gaussian_mechanism_outputs"

    # system size / structure
    n_blocks: int = 12  # total dimension = 2 * n_blocks
    base_alpha_min: float = 0.08
    base_alpha_max: float = 0.95
    weight_decay: float = 0.9

    # exact curves over global scale t in [0, 1]
    scale_min: float = 0.05
    scale_max: float = 1.0
    n_scale_exact: int = 250

    # finite-sample imitation over a coarser scale grid
    n_scale_finite: int = 18

    # lambda sweeps at fixed scales
    lambda_min: float = 0.0
    lambda_max: float = 0.95
    n_lambda_exact: int = 220
    lambda_sweep_scales: tuple[float, ...] = (0.2, 0.9)

    # local derivative wrt lambda at lambda0
    lambda0: float = 0.4
    lambda_fd_delta: float = 0.02

    # stochastic imitation parameters
    sigma: float = 0.2
    n_values: tuple[int, ...] = (4, 12, 64)
    n_seeds: int = 40
    n_iters: int = 100
    burn_in: int = 80
    update_mode: str = "async_theta_first"  # or "synchronous"

    # block contribution snapshots
    contribution_scales: float = 0.9

    # plotting
    dpi: int = 560
    grid_alpha: float = 0.22
    lw_main: float = 2.2
    lw_aux: float = 1.8
    marker_size: float = 5.0

    # reward parameters
    eta_p: float = 0.18
    eta_q: float = 0.22

    lambda_zoom_halfwidth: float = 0.4
    master_figsize: tuple[float, float] = (16, 26)
    master_hspace: float = 0.42
    master_wspace: float = 0.28

    lambda_empirical_n_values: tuple[int, ...] = (4, 32)
    lambda_empirical_n_seeds: int=20


class BlockDiagonalToySystem:
    """High-dimensional block-diagonal coupled system.

    Let d = 2m. For global coupling scale t in [0,1], define
        A(t) = blkdiag(t * beta_1 * R_1, ..., t * beta_m * R_m)

    The coupled fixed-point equations are
        F_p(theta, phi, lambda) = theta - phi - lambda a = 0
        F_q(theta, phi, lambda) = phi - A(t) theta = 0

    Stable point:
        theta* = (I - A(t))^{-1} lambda a
        phi*   = A(t) theta*

    Reward directions:
        J_p(theta) = g_p^T theta - eta_p / 2 ||theta||^2  (eta_p can be 0 for pure linear rewards)
        J_q(phi)   = g_q^T phi - eta_q / 2 ||phi||^2    (eta_q can be 0 for pure linear rewards)

    Theory objects:
        S_p = S_q = (I - A)^{-1}
        C_q = A

    Alignment proxies:
        rho_p = cos(g_p, a)
        rho_q = cos(g_q, A a)

    Exact local derivatives:
        dJ_p/dlambda = g_p^T S_p a
        dJ_q/dlambda = g_q^T S_q A a
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.n_blocks = cfg.n_blocks
        self.dim = 2 * self.n_blocks

        def rot2d(deg: float) -> Array:
            rad = np.deg2rad(deg)
            c, s = np.cos(rad), np.sin(rad)
            return np.array([[c, -s], [s, c]], dtype=float)

        np.random.seed(2026330)
        angles_R = np.random.permutation(np.linspace(5.0, 85.0, self.n_blocks // 2).tolist() + \
           (-np.linspace(10.0, 80.0, self.n_blocks - self.n_blocks // 2)).tolist())
        self.angles_R = angles_R
        self.R_blocks = [rot2d(float(deg)) for deg in angles_R]

        # heterogeneous block strengths beta_i in [base_alpha_min, base_alpha_max]
        np.random.seed(1999)
        self.base_alphas = np.random.uniform(low=cfg.base_alpha_min, high=cfg.base_alpha_max, size=self.n_blocks)

        # structured block-wise directions and weights
        a_blocks = []
        gp_blocks = []
        gq_blocks = []
        self.block_weights = []
        a_base = np.array([1.0, 2.0], dtype=float)
        gp_base = np.array([1.0, 0.0], dtype=float)
        gq_base = np.array([1.0, 4.0], dtype=float)

        # structured heterogeneous angles (degrees)
        angles_a = np.linspace(-70.0, 70.0, self.n_blocks)
        angles_gp = np.linspace(25.0, -55.0, self.n_blocks)
        angles_gq = np.linspace(-40.0, 60.0, self.n_blocks)
        # optional mild magnitude heterogeneity
        scale_a = 1.0 + 0.20 * np.sin(np.linspace(0.0, 2.0 * np.pi, self.n_blocks))
        scale_gp = 1.0 + 0.15 * np.cos(np.linspace(0.0, 1.5 * np.pi, self.n_blocks))
        scale_gq = 1.0 + 0.18 * np.sin(np.linspace(0.3, 1.8 * np.pi, self.n_blocks))
        self.angles_a = angles_a
        self.angles_gp = angles_gp
        self.angles_gq = angles_gq

        for i in range(self.n_blocks):
            w = cfg.weight_decay ** i
            self.block_weights.append(w)

            Qa = rot2d(float(angles_a[i]))
            Qp = rot2d(float(angles_gp[i]))
            Qq = rot2d(float(angles_gq[i]))

            a_i = w * scale_a[i] * (Qa @ a_base)
            gp_i = w * scale_gp[i] * (Qp @ gp_base)
            gq_i = w * scale_gq[i] * (Qq @ gq_base)

            a_blocks.append(a_i)
            gp_blocks.append(gp_i)
            gq_blocks.append(gq_i)
        
        np.random.seed(1108)
        self.block_weights = np.random.permutation(np.array(self.block_weights))
        self.a = np.concatenate(np.random.permutation(a_blocks))
        self.g_p = np.concatenate(np.random.permutation(gp_blocks))
        self.g_q = np.concatenate(np.random.permutation(gq_blocks))

    # ---------- Block algebra ----------
    def block_alpha(self, scale: float, idx: int) -> float:
        return float(scale * self.base_alphas[idx])

    def block_matrix_A(self, scale: float, idx: int) -> Array:
        return self.block_alpha(scale, idx) * self.R_blocks[idx]

    def block_matrix_S(self, scale: float, idx: int) -> Array:
        A_i = self.block_matrix_A(scale, idx)
        return np.linalg.inv(np.eye(2) - A_i)

    def apply_blockdiag(self, scale: float, vec: Array, kind: str) -> Array:
        out = np.zeros_like(vec)
        for i in range(self.n_blocks):
            sl = slice(2 * i, 2 * i + 2)
            block = vec[sl]
            if kind == "A":
                out[sl] = self.block_matrix_A(scale, i) @ block
            elif kind == "S":
                out[sl] = self.block_matrix_S(scale, i) @ block
            else:
                raise ValueError(f"Unknown kind={kind}")
        return out

    def A_apply(self, scale: float, vec: Array) -> Array:
        return self.apply_blockdiag(scale, vec, kind="A")

    def S_apply(self, scale: float, vec: Array) -> Array:
        return self.apply_blockdiag(scale, vec, kind="S")

    # ---------- Fixed point / rewards ----------
    def theta_star(self, scale: float, lam: float) -> Array:
        return lam * self.S_apply(scale, self.a)

    def phi_star(self, scale: float, lam: float) -> Array:
        return self.A_apply(scale, self.theta_star(scale, lam))

    def J_p(self, theta: Array) -> float:
        return float(self.g_p @ theta - 0.5 * self.cfg.eta_p * (theta @ theta))

    def J_q(self, phi: Array) -> float:
        return float(self.g_q @ phi - 0.5 * self.cfg.eta_q * (phi @ phi))

    def grad_J_p(self, theta: Array) -> Array:
        return self.g_p - self.cfg.eta_p * theta

    def grad_J_q(self, phi: Array) -> Array:
        return self.g_q - self.cfg.eta_q * phi

    def J_p_star(self, scale: float, lam: float) -> float:
        return self.J_p(self.theta_star(scale, lam))

    def J_q_star(self, scale: float, lam: float) -> float:
        return self.J_q(self.phi_star(scale, lam))

    # ---------- Alignment proxies ----------
    @staticmethod
    def cosine(u: Array, v: Array) -> float:
        nu = float(np.linalg.norm(u))
        nv = float(np.linalg.norm(v))
        if nu == 0.0 or nv == 0.0:
            return 0.0
        return float((u @ v) / (nu * nv))

    def rho_p(self, scale: float, lam: float) -> float:
        theta = self.theta_star(scale, lam)
        return self.cosine(self.grad_J_p(theta), theta)

    def rho_q(self, scale: float, lam: float) -> float:
        phi = self.phi_star(scale, lam)
        return self.cosine(self.grad_J_q(phi), self.A_apply(scale, self.a))

    # ---------- Exact derivatives ----------
    def dJp_dlambda_explicit(self, scale: float, lam: float) -> float:
        theta = self.theta_star(scale, lam)
        dtheta_dlam = self.S_apply(scale, self.a)
        return float(self.grad_J_p(theta) @ dtheta_dlam)

    def dJq_dlambda_explicit(self, scale: float, lam: float) -> float:
        phi = self.phi_star(scale, lam)
        dtheta_dlam = self.S_apply(scale, self.a)
        dphi_dlam = self.A_apply(scale, dtheta_dlam)
        return float(self.grad_J_q(phi) @ dphi_dlam)

    # ---------- Theorem RHS forms ----------
    def theorem_rhs_p(self, scale: float, lam: float) -> float:
        theta = self.theta_star(scale, lam)
        phi = self.phi_star(scale, lam)
        mean_curated = phi + self.a
        m_h = mean_curated - theta  # E[z - theta*] under curated component
        return float((self.grad_J_p(theta) @ self.S_apply(scale, m_h)) / (1.0 - lam))

    def theorem_rhs_q(self, scale: float, lam: float) -> float:
        theta = self.theta_star(scale, lam)
        phi = self.phi_star(scale, lam)
        mean_curated = phi + self.a
        m_h = mean_curated - theta
        return float((self.grad_J_q(phi) @ self.S_apply(scale, self.A_apply(scale, m_h))) / (1.0 - lam))

    # ---------- Exact finite differences ----------
    def finite_difference_p(self, scale: float, lam: float, delta: float) -> float:
        theta_plus = self.theta_star(scale, lam + delta)
        theta_minus = self.theta_star(scale, lam - delta)
        return (self.J_p(theta_plus) - self.J_p(theta_minus)) / (2.0 * delta)

    def finite_difference_q(self, scale: float, lam: float, delta: float) -> float:
        phi_plus = self.phi_star(scale, lam + delta)
        phi_minus = self.phi_star(scale, lam - delta)
        return (self.J_q(phi_plus) - self.J_q(phi_minus)) / (2.0 * delta)

    def local_delta_signs(self, scale: float, lam: float, delta: float) -> dict[str, float]:
        jp_minus = self.J_p_star(scale, lam - delta)
        jp_mid = self.J_p_star(scale, lam)
        jp_plus = self.J_p_star(scale, lam + delta)
        jq_minus = self.J_q_star(scale, lam - delta)
        jq_mid = self.J_q_star(scale, lam)
        jq_plus = self.J_q_star(scale, lam + delta)
        return {
            "jp_delta": jp_plus - jp_minus,
            "jq_delta": jq_plus - jq_minus,
            "jp_forward": jp_plus - jp_mid,
            "jq_forward": jq_plus - jq_mid,
        }

    # ---------- Block-wise contributions ----------
    def block_contributions(self, scale: float) -> dict[str, Array]:
        raw_p = np.zeros(self.n_blocks)
        distorted_p = np.zeros(self.n_blocks)
        raw_q = np.zeros(self.n_blocks)
        distorted_q = np.zeros(self.n_blocks)
        for i in range(self.n_blocks):
            sl = slice(2 * i, 2 * i + 2)
            ai = self.a[sl]
            gpi = self.g_p[sl]
            gqi = self.g_q[sl]
            Ai = self.block_matrix_A(scale, i)
            Si = self.block_matrix_S(scale, i)
            raw_p[i] = float(gpi @ ai)
            distorted_p[i] = float(gpi @ (Si @ ai))
            raw_q[i] = float(gqi @ (Ai @ ai))
            distorted_q[i] = float(gqi @ (Si @ (Ai @ ai)))
        return {
            "raw_p": raw_p,
            "distorted_p": distorted_p,
            "raw_q": raw_q,
            "distorted_q": distorted_q,
        }

    # ---------- Stochastic imitation ----------
    def simulate_run(
        self,
        rng: np.random.Generator,
        scale: float,
        lam: float,
        n: int,
        n_iters: int,
        burn_in: int,
        sigma: float,
        update_mode: str,
    ) -> tuple[float, float]:
        theta = np.zeros(self.dim)
        phi = np.zeros(self.dim)
        jp_hist = []
        jq_hist = []
        noise_scale = sigma / math.sqrt(n)

        for t in range(n_iters):
            # p-side empirical mixture mean: phi + (k/n) a + Gaussian mean noise
            k = rng.binomial(n=n, p=lam)
            eps_p = rng.normal(loc=0.0, scale=noise_scale, size=self.dim)
            theta_next = phi + (k / float(n)) * self.a + eps_p

            # q-side update uses current or newly updated theta, depending on mode
            theta_for_q = theta_next if update_mode == "async_theta_first" else theta
            eps_q = rng.normal(loc=0.0, scale=noise_scale, size=self.dim)
            phi_next = self.A_apply(scale, theta_for_q) + eps_q

            theta, phi = theta_next, phi_next
            if t >= burn_in:
                jp_hist.append(self.J_p(theta))
                jq_hist.append(self.J_q(phi))

        return float(np.mean(jp_hist)), float(np.mean(jq_hist))

    def empirical_central_difference(
        self,
        rng: np.random.Generator,
        scale: float,
        lam0: float,
        delta: float,
        n: int,
        n_iters: int,
        burn_in: int,
        sigma: float,
        update_mode: str,
    ) -> tuple[float, float]:
        jp_plus, jq_plus = self.simulate_run(
            rng, scale, lam0 + delta, n, n_iters, burn_in, sigma, update_mode
        )
        jp_minus, jq_minus = self.simulate_run(
            rng, scale, lam0 - delta, n, n_iters, burn_in, sigma, update_mode
        )
        d_jp = (jp_plus - jp_minus) / (2.0 * delta)
        d_jq = (jq_plus - jq_minus) / (2.0 * delta)
        return d_jp, d_jq


# ---------- Plotting / reporting ----------
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(fig: plt.Figure, path: Path, dpi: int) -> None:
    fig.tight_layout(pad=1.05)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")


def style_axis(ax: plt.Axes, cfg: Config, xlabel: str | None = None, ylabel: str | None = None) -> None:
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=cfg.grid_alpha, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def zero_crossings(x: Array, y: Array) -> list[float]:
    out = []
    for i in range(len(x) - 1):
        if y[i] == 0.0:
            out.append(float(x[i]))
        elif y[i] * y[i + 1] < 0:
            t = abs(y[i]) / (abs(y[i]) + abs(y[i + 1]))
            out.append(float((1 - t) * x[i] + t * x[i + 1]))
    return out

def compute_exact_fd_data(system: BlockDiagonalToySystem, cfg: Config):
    scales = np.linspace(cfg.scale_min, cfg.scale_max, cfg.n_scale_exact)

    d_jp_explicit = np.array([system.dJp_dlambda_explicit(s, cfg.lambda0) for s in scales])
    d_jp_theorem = np.array([system.theorem_rhs_p(s, cfg.lambda0) for s in scales])
    d_jp_fd = np.array([system.finite_difference_p(s, cfg.lambda0, cfg.lambda_fd_delta) for s in scales])

    d_jq_explicit = np.array([system.dJq_dlambda_explicit(s, cfg.lambda0) for s in scales])
    d_jq_theorem = np.array([system.theorem_rhs_q(s, cfg.lambda0) for s in scales])
    d_jq_fd = np.array([system.finite_difference_q(s, cfg.lambda0, cfg.lambda_fd_delta) for s in scales])

    summary = {
        "max_abs_Jp_explicit_minus_theorem": float(np.max(np.abs(d_jp_explicit - d_jp_theorem))),
        "max_abs_Jp_explicit_minus_fd": float(np.max(np.abs(d_jp_explicit - d_jp_fd))),
        "max_abs_Jq_explicit_minus_theorem": float(np.max(np.abs(d_jq_explicit - d_jq_theorem))),
        "max_abs_Jq_explicit_minus_fd": float(np.max(np.abs(d_jq_explicit - d_jq_fd))),
    }

    return {
        "scales": scales,
        "d_jp_explicit": d_jp_explicit,
        "d_jp_theorem": d_jp_theorem,
        "d_jp_fd": d_jp_fd,
        "d_jq_explicit": d_jq_explicit,
        "d_jq_theorem": d_jq_theorem,
        "d_jq_fd": d_jq_fd,
        "summary": summary,
    }


def compute_finite_sample_data(system: BlockDiagonalToySystem, cfg: Config):
    scales = np.linspace(cfg.scale_min, cfg.scale_max, cfg.n_scale_finite)
    exact_jp = np.array([system.dJp_dlambda_explicit(s, cfg.lambda0) for s in scales])
    exact_jq = np.array([system.dJq_dlambda_explicit(s, cfg.lambda0) for s in scales])

    all_results_jp: dict[int, Array] = {}
    all_results_jq: dict[int, Array] = {}

    master_rng = np.random.default_rng(12345)
    for n in cfg.n_values:
        jp_mat = np.zeros((cfg.n_seeds, len(scales)))
        jq_mat = np.zeros((cfg.n_seeds, len(scales)))
        for seed_idx in range(cfg.n_seeds):
            rng = np.random.default_rng(master_rng.integers(1, 10**9))
            for j, scale in enumerate(scales):
                d_jp, d_jq = system.empirical_central_difference(
                    rng=rng,
                    scale=scale,
                    lam0=cfg.lambda0,
                    delta=cfg.lambda_fd_delta,
                    n=n,
                    n_iters=cfg.n_iters,
                    burn_in=cfg.burn_in,
                    sigma=cfg.sigma,
                    update_mode=cfg.update_mode,
                )
                jp_mat[seed_idx, j] = d_jp
                jq_mat[seed_idx, j] = d_jq
        all_results_jp[n] = jp_mat
        all_results_jq[n] = jq_mat

    return {
        "scales": scales,
        "exact_jp": exact_jp,
        "exact_jq": exact_jq,
        "all_results_jp": all_results_jp,
        "all_results_jq": all_results_jq,
    }


def compute_lambda_sweep_empirical_data(system: BlockDiagonalToySystem, cfg: Config):
    lambdas_full = np.linspace(cfg.lambda_min, cfg.lambda_max, cfg.n_lambda_exact)

    lam_lo = max(cfg.lambda_min, cfg.lambda0 - cfg.lambda_zoom_halfwidth)
    lam_hi = min(cfg.lambda_max, cfg.lambda0 + cfg.lambda_zoom_halfwidth)
    mask = (lambdas_full >= lam_lo) & (lambdas_full <= lam_hi)
    lambdas = lambdas_full[mask]

    results: dict[float, dict[int, dict[str, Array]]] = {}
    master_rng = np.random.default_rng(20250330)

    for scale in cfg.lambda_sweep_scales:
        results[scale] = {}
        for n in cfg.lambda_empirical_n_values:
            jp_mat = np.zeros((cfg.lambda_empirical_n_seeds, len(lambdas)))
            jq_mat = np.zeros((cfg.lambda_empirical_n_seeds, len(lambdas)))

            for seed_idx in range(cfg.lambda_empirical_n_seeds):
                rng = np.random.default_rng(master_rng.integers(1, 10**9))
                for j, lam in enumerate(lambdas):
                    jp_hat, jq_hat = system.simulate_run(
                        rng=rng,
                        scale=scale,
                        lam=lam,
                        n=n,
                        n_iters=cfg.n_iters,
                        burn_in=cfg.burn_in,
                        sigma=cfg.sigma,
                        update_mode=cfg.update_mode,
                    )
                    jp_mat[seed_idx, j] = jp_hat
                    jq_mat[seed_idx, j] = jq_hat

            jp_mean = np.mean(jp_mat, axis=0)
            jq_mean = np.mean(jq_mat, axis=0)

            jp_se = np.std(jp_mat, axis=0, ddof=1) / np.sqrt(cfg.lambda_empirical_n_seeds)
            jq_se = np.std(jq_mat, axis=0, ddof=1) / np.sqrt(cfg.lambda_empirical_n_seeds)

            results[scale][n] = {
                "lambdas": lambdas,
                "jp_mean": jp_mean,
                "jp_lo": jp_mean - 1.96 * jp_se,
                "jp_hi": jp_mean + 1.96 * jp_se,
                "jq_mean": jq_mean,
                "jq_lo": jq_mean - 1.96 * jq_se,
                "jq_hi": jq_mean + 1.96 * jq_se,
            }

    return results

def plot_master_figure(system: BlockDiagonalToySystem, cfg: Config, outdir: Path):
    fd_data = compute_exact_fd_data(system, cfg)
    finite_data = compute_finite_sample_data(system, cfg)

    fig, axes = plt.subplots(4, 2, figsize=(16, 20))
    plt.subplots_adjust(hspace=1, wspace=0.4)

    # ---------- Row 1: figure 1 ----------
    scales = np.linspace(cfg.scale_min, cfg.scale_max, cfg.n_scale_exact)
    rho_p = np.array([system.rho_p(s, cfg.lambda0) for s in scales])
    rho_q = np.array([system.rho_q(s, cfg.lambda0) for s in scales])

    # (1,1) alignment proxy
    ax = axes[0, 0]
    ax.plot(scales, rho_p, linewidth=cfg.lw_main, label=r"$\rho_p$")
    ax.plot(scales, rho_q, linewidth=cfg.lw_main, label=r"$\rho_q$")
    style_axis(ax, cfg, xlabel="global coupling scale t", ylabel="alignment proxy")
    ax.set_title("Alignment proxy")
    ax.legend(frameon=False, ncol=2)

    # (1,2) derivative overview + exact consistency merged
    ax = axes[0, 1]
    # J_p: explicit / theorem / finite diff
    ax.plot(fd_data["scales"], fd_data["d_jp_explicit"],
            linewidth=cfg.lw_main,
            label=r"$\partial J_p/\partial\lambda$ (explicit)")
    ax.plot(fd_data["scales"], fd_data["d_jp_theorem"],
            linestyle="--", linewidth=cfg.lw_aux,
            label=r"$\partial J_p/\partial\lambda$ (theorem)")

    # J_q: explicit / theorem / finite diff
    ax.plot(fd_data["scales"], fd_data["d_jq_explicit"],
            linewidth=cfg.lw_main,
            label=r"$\partial J_q/\partial\lambda$ (explicit)")
    ax.plot(fd_data["scales"], fd_data["d_jq_theorem"],
            linestyle="--", linewidth=cfg.lw_aux,
            label=r"$\partial J_q/\partial\lambda$ (theorem)")
    
    ax.axhline(0.0, linewidth=1.0)

    for xc in zero_crossings(fd_data["scales"], fd_data["d_jp_explicit"]):
        ax.axvline(xc, linestyle="--", linewidth=1.0, alpha=0.45)
    for xc in zero_crossings(fd_data["scales"], fd_data["d_jq_explicit"]):
        ax.axvline(xc, linestyle=":", linewidth=1.0, alpha=0.45)

    style_axis(ax, cfg, xlabel="global coupling scale t", ylabel="local derivative")
    ax.set_title("Derivative overview")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)

    # ---------- Row 2: finite-sample imitation ----------
    scales_f = finite_data["scales"]

    ax = axes[1, 0]
    ax.plot(scales_f, finite_data["exact_jp"], linewidth=cfg.lw_main, label="exact theory")
    for n in cfg.n_values:
        mean, lo, hi = mean_and_ci(finite_data["all_results_jp"][n])
        ax.plot(scales_f, mean, linewidth=cfg.lw_aux, label=f"empirical n={n}")
        ax.fill_between(scales_f, lo, hi, alpha=0.16)
    ax.axhline(0.0, linewidth=1.0)
    style_axis(ax, cfg, xlabel="global coupling scale t", ylabel=r"$\partial J_p / \partial \lambda$")
    ax.set_title("Finite-sample imitation: $J_p$")

    ax = axes[1, 1]
    ax.plot(scales_f, finite_data["exact_jq"], linewidth=cfg.lw_main, label="exact theory")
    for n in cfg.n_values:
        mean, lo, hi = mean_and_ci(finite_data["all_results_jq"][n])
        ax.plot(scales_f, mean, linewidth=cfg.lw_aux, label=f"empirical n={n}")
        ax.fill_between(scales_f, lo, hi, alpha=0.16)
    ax.axhline(0.0, linewidth=1.0)
    style_axis(ax, cfg, xlabel="global coupling scale t", ylabel=r"$\partial J_q / \partial \lambda$")
    ax.set_title("Finite-sample imitation: $J_q$")

    # ---------- Row 3: block contributions ----------
    row_idx = 2
    scale = cfg.contribution_scales
    contrib = system.block_contributions(scale)
    x = np.arange(system.n_blocks)
    width = 0.42

    ax = axes[row_idx, 0]
    ax.bar(x - width / 2, contrib["raw_p"], width=width, label="raw self-alignment")
    ax.bar(x + width / 2, contrib["distorted_p"], width=width, label="distorted self-effect")
    ax.axhline(0.0, linewidth=1.0)
    style_axis(ax, cfg, xlabel="block index", ylabel="block contribution")
    ax.set_title(fr"$J_p$ block contributions at $t={scale:.2f}$")
    ax.legend(frameon=False, fontsize=8.0)

    ax = axes[row_idx, 1]
    ax.bar(x - width / 2, contrib["raw_q"], width=width, label="raw cross-alignment")
    ax.bar(x + width / 2, contrib["distorted_q"], width=width, label="distorted cross-effect")
    ax.axhline(0.0, linewidth=1.0)
    style_axis(ax, cfg, xlabel="block index", ylabel="block contribution")
    ax.set_title(fr"$J_q$ block contributions at $t={scale:.2f}$")
    ax.legend(frameon=False, fontsize=8.0)

    # ---------- Row 4: lambda sweeps ----------
    lambdas_full = np.linspace(cfg.lambda_min, cfg.lambda_max, cfg.n_lambda_exact)
    lam_lo = max(cfg.lambda_min, cfg.lambda0 - cfg.lambda_zoom_halfwidth)
    lam_hi = min(cfg.lambda_max, cfg.lambda0 + cfg.lambda_zoom_halfwidth)
    mask = (lambdas_full >= lam_lo) & (lambdas_full <= lam_hi)
    lambdas = lambdas_full[mask]

    empirical_data = compute_lambda_sweep_empirical_data(system, cfg)

    axp = axes[3, 0]
    axq = axes[3, 1]

    for scale in cfg.lambda_sweep_scales:
        jp_vals = np.array([system.J_p_star(scale, lam) for lam in lambdas])
        jq_vals = np.array([system.J_q_star(scale, lam) for lam in lambdas])

        axp.plot(lambdas, jp_vals, linewidth=cfg.lw_main, label=fr"exact, $t={scale:.2f}$")
    
        axq.plot(lambdas, jq_vals, linewidth=cfg.lw_main, label=fr"exact, $t={scale:.2f}$")

        for n in cfg.lambda_empirical_n_values:
            emp = empirical_data[scale][n]

            axp.plot(
                emp["lambdas"],
                emp["jp_mean"],
                linewidth=cfg.lw_aux,
                linestyle=":",
                label=fr"empirical, $t={scale:.2f},\,n={n}$"
            )
            axp.fill_between(
                emp["lambdas"],
                emp["jp_lo"],
                emp["jp_hi"],
                alpha=0.12
            )

            axq.plot(
                emp["lambdas"],
                emp["jq_mean"],
                linewidth=cfg.lw_aux,
                linestyle=":",
                label=fr"empirical, $t={scale:.2f},\,n={n}$"
            )
            axq.fill_between(
                emp["lambdas"],
                emp["jq_lo"],
                emp["jq_hi"],
                alpha=0.12
            )

    axp.axvline(cfg.lambda0, linewidth=1.0, alpha=0.55)
    axq.axvline(cfg.lambda0, linewidth=1.0, alpha=0.55)

    style_axis(axp, cfg, xlabel=r"curation ratio $\lambda$", ylabel=r"$J_p$ after convergence")
    style_axis(axq, cfg, xlabel=r"curation ratio $\lambda$", ylabel=r"$J_q$ after convergence")
    axp.set_title(r"$J_p$ vs $\lambda$ around $\lambda_0$")
    axq.set_title(r"$J_q$ vs $\lambda$ around $\lambda_0$")
    axp.legend(frameon=False, fontsize=8.0)
    axq.legend(frameon=False, fontsize=8.0)

    savefig(fig, outdir / "master_figure_hd.png", cfg.dpi)
    plt.savefig(outdir / "master_figure_hd.svg", format="svg", dpi=cfg.dpi)
    plt.close(fig)

    # keep summaries
    with open(outdir / "exact_consistency_summary_hd.txt", "w", encoding="utf-8") as f:
        for k, v in fd_data["summary"].items():
            f.write(f"{k}: {v:.12e}\n")

def mean_and_ci(arr: Array) -> tuple[Array, Array, Array]:
    mean = np.mean(arr, axis=0)
    sem = np.std(arr, axis=0, ddof=1) / math.sqrt(arr.shape[0])
    lo = mean - 1.96 * sem
    hi = mean + 1.96 * sem
    return mean, lo, hi


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="High-dimensional mechanism gaussian experiment")
    parser.add_argument("--output_dir", type=str, default=Config.output_dir)
    parser.add_argument("--n_blocks", type=int, default=Config.n_blocks)
    parser.add_argument("--n_scale_exact", type=int, default=Config.n_scale_exact)
    parser.add_argument("--n_scale_finite", type=int, default=Config.n_scale_finite)
    parser.add_argument("--n_lambda_exact", type=int, default=Config.n_lambda_exact)
    parser.add_argument("--n_seeds", type=int, default=Config.n_seeds)
    parser.add_argument("--n_iters", type=int, default=Config.n_iters)
    parser.add_argument("--burn_in", type=int, default=Config.burn_in)
    parser.add_argument("--sigma", type=float, default=Config.sigma)
    parser.add_argument("--lambda0", type=float, default=Config.lambda0)
    parser.add_argument("--lambda_fd_delta", type=float, default=Config.lambda_fd_delta)
    parser.add_argument("--update_mode", type=str, choices=["async_theta_first", "synchronous"], default=Config.update_mode)
    return parser.parse_args()


def build_config_from_args(args: argparse.Namespace) -> Config:
    return Config(
        output_dir=args.output_dir,
        n_blocks=args.n_blocks,
        n_scale_exact=args.n_scale_exact,
        n_scale_finite=args.n_scale_finite,
        n_lambda_exact=args.n_lambda_exact,
        n_seeds=args.n_seeds,
        n_iters=args.n_iters,
        burn_in=args.burn_in,
        sigma=args.sigma,
        lambda0=args.lambda0,
        lambda_fd_delta=args.lambda_fd_delta,
        update_mode=args.update_mode,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config_from_args(args)
    outdir = ensure_dir(cfg.output_dir)
    system = BlockDiagonalToySystem(cfg)

    plot_master_figure(system, cfg, outdir)

    print(f"Done. Outputs written to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
