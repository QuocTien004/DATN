#!/usr/bin/env python
"""Thực nghiệm Huấn luyện Dài hạn (Long RL Training Experiment).

Script thiết lập độc lập cho phép người dùng tự chạy thử nghiệm huấn luyện
ở quy mô tương đương hoặc vượt qua các đợt train dài nhất lịch sử dự án:
- Đợt train 400.000 bước môi trường (11 tiếng qua đêm vào 19-20/9/2026).
- Đợt train 30.000 bước môi trường (750 AC updates vào 10/10/2026).

Hỗ trợ 2 phương thức tương tác (--interaction):
1. offline: Huấn luyện trực tiếp Actor-Critic qua tưởng tượng trên Replay Buffer
   (Siêu tốc: 10.000 updates chỉ mất ~1.5 - 2 tiếng thay vì 11 tiếng render).
2. online: Xe lăn bánh thật trong môi trường MetaDrive suốt N bước (ví dụ 400.000 bước,
   thu thập transitions mới, cập nhật 25 AC updates mỗi 1.000 bước môi trường).

Hỗ trợ 3 chế độ thuật toán (--mode):
1. raw: Thuần RL nguyên bản (đối chứng trực tiếp với đợt 30k/400k cũ).
2. safe: Bọc P3 Safe Wrapper (phạt -20 điểm va chạm, cắt lệch làn > 0.28).
3. constrained: P3 Safe + Phạt đạp lút chân ga khi throttle > 0.4.

Toàn bộ kết quả, log và checkpoint được lưu riêng biệt vào:
- Checkpoints: checkpoints/exp_long_test/
- Logs: logs/exp_long_test/
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from envs.metadrive_wrapper import MetaDriveImageEnv, make_env
from evaluation.evaluate import LatentActorPolicy
from evaluation.p1_p4_diagnostics import boundary_predictions
from models.actor_critic import ActorCritic
from models.world_model import build_world_model, load_world_model_state
from training.batches import replay_batch_to_torch
from training.train_agent import (
    posterior_start_states,
    imagine_rollout,
    freeze_parameters,
    _continuation_weights,
    _validate_start_states,
)
from utils.replay_buffer import ReplayBuffer, Transition

PAIRED_BENCHMARK = [
    {"map": 44200, "seed": 44200 + 1234},
    {"map": 44200, "seed": 44200 + 2234},
    {"map": 44205, "seed": 44205 + 1234},
    {"map": 44205, "seed": 44205 + 2234},
    {"map": 44500, "seed": 44500 + 1234},
    {"map": 44500, "seed": 44500 + 2234},
    {"map": 44501, "seed": 44501 + 1234},
    {"map": 44501, "seed": 44501 + 2234},
]


class P3SafeWrappedPredictor(nn.Module):
    """P3 Safe Wrapper: Phạt va chạm thực tế -20 điểm và triệt tiêu thưởng phóng nhanh cua gắt."""

    def __init__(self, base_reward: nn.Module, base_continue: nn.Module, rssm_continuous: nn.Module, target_key: str):
        super().__init__()
        self.base_reward = base_reward
        self.base_continue = base_continue
        self.rssm_continuous = rssm_continuous
        self.target_key = target_key
        self.continuous_state_dim = 19

    def forward(self, h: torch.Tensor, z: torch.Tensor, *, c: torch.Tensor | None = None) -> torch.Tensor:
        r_sym = self.base_reward(h, z)
        c_logit = self.base_continue(h, z)
        if c is not None and self.rssm_continuous is not None:
            phys_s = self.rssm_continuous.decode(c)
            # 1. Phạt va chạm -20 (symlog -3.04) với ngưỡng clearance 0.095
            p3 = boundary_predictions(
                phys_s, r_sym, c_logit,
                threshold=0.095,
                alpha=0.18,
                sharpness=120.0,
                speed_threshold=0.05,
                penalty_symlog=-3.04,
            )
            r_out = p3["reward"]
            c_out = p3["continue"]

            # 2. Phạt tốc độ khi ôm cua / lệch tâm
            lat_dev = (phys_s[:, 2:3] - 0.5).abs()
            speed = phys_s[:, 3:4]
            excess_speed_in_turn = torch.relu(speed - 0.35) * torch.relu(lat_dev - 0.12) * 8.0
            r_out = r_out - excess_speed_in_turn

            # 3. Kẹp logit sống sót nếu xe văng quá rộng (> 0.28)
            extreme_dev = lat_dev > 0.28
            c_out = torch.where(extreme_dev, torch.minimum(c_out, torch.full_like(c_out, -4.0)), c_out)

            return r_out if self.target_key == "reward" else c_out
        return r_sym if self.target_key == "reward" else c_logit


def evaluate_paired_benchmark(policy: LatentActorPolicy, env_cfg_base: dict, device: torch.device) -> dict:
    """Đánh giá chính sách trên đúng 8 kịch bản Paired Benchmark chuẩn."""
    routes = []
    finishes = 0
    outs = 0
    crashes = 0
    rewards = []
    throttles = []
    steerings = []

    for item in PAIRED_BENCHMARK:
        cfg = dict(env_cfg_base)
        cfg["start_seed"] = item["seed"]
        cfg["num_scenarios"] = 1
        cfg["horizon"] = 1000

        env = make_env(cfg)
        obs, _ = env.reset()
        policy.reset()

        ep_r = 0.0
        done = False
        final_info = {}
        ep_actions = []

        while not done:
            action = policy(obs)
            ep_actions.append(action)
            obs, r, term, trunc, info = env.step(action)
            ep_r += r
            done = term or trunc
            final_info = info

        env.close()

        acts = np.array(ep_actions)
        if len(acts) > 0:
            steerings.append(float(np.mean(acts[:, 0])))
            throttles.append(float(np.mean(acts[:, 1])))

        route_comp = final_info.get("route_completion", 0.0) * 100.0
        routes.append(route_comp)
        rewards.append(ep_r)
        if final_info.get("arrive_dest", False):
            finishes += 1
        elif final_info.get("crash", False):
            crashes += 1
        elif final_info.get("out_of_road", False):
            outs += 1

    return {
        "route_mean": float(np.mean(routes)),
        "route_std": float(np.std(routes)),
        "finish": finishes,
        "out": outs,
        "crash": crashes,
        "reward_mean": float(np.mean(rewards)),
        "throttle_mean": float(np.mean(throttles)) if throttles else 0.0,
        "steering_mean": float(np.mean(steerings)) if steerings else 0.0,
    }


def plot_training_curves(history: list[dict], save_path: Path) -> None:
    """Tự động vẽ và cập nhật biểu đồ tiến trình huấn luyện."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if len(history) < 1:
        return

    steps = [h.get("step", h.get("update", h.get("env_step", 0))) for h in history]
    routes = [h["eval"]["route_mean"] for h in history]
    finishes = [h["eval"]["finish"] for h in history]
    outs = [h["eval"]["out"] for h in history]
    crashes = [h["eval"]["crash"] for h in history]
    throttles = [h["eval"].get("throttle_mean", 0.0) for h in history]
    steerings = [h["eval"].get("steering_mean", 0.0) for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Tiến Trình Huấn Luyện RL (Training Progress Curves)", fontsize=15, fontweight="bold")

    # 1. Quãng đường hoàn thành
    ax = axes[0, 0]
    ax.plot(steps, routes, marker="o", color="#2B6CB0", linewidth=2.5, label="Route Completion (%)")
    ax.axhline(57.24, color="gray", linestyle="--", label="BC Baseline cũ (57.24%)")
    ax.axhline(71.78, color="green", linestyle=":", label="Update 0 Baseline (71.78%)")
    ax.set_title("Quãng đường hoàn thành (Route %)")
    ax.set_xlabel("Steps / Updates")
    ax.set_ylabel("Route Completion (%)")
    ax.set_ylim(-5, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    # 2. Kết quả kịch bản (Finish, Out, Crash)
    ax = axes[0, 1]
    ax.plot(steps, finishes, marker="s", color="#38A169", linewidth=2, label="Về đích (Finish)")
    ax.plot(steps, outs, marker="^", color="#DD6B20", linewidth=2, label="Văng đường (Out)")
    ax.plot(steps, crashes, marker="x", color="#E53E3E", linewidth=2, label="Va chạm (Crash)")
    ax.set_title("Kết quả 8 kịch bản Paired Benchmark")
    ax.set_xlabel("Steps / Updates")
    ax.set_ylabel("Số lượng ca (trên 8)")
    ax.set_ylim(-0.5, 8.5)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. Hành vi điều khiển (Chân ga vs Góc lái)
    ax = axes[1, 0]
    ax.plot(steps, throttles, marker="d", color="#3182CE", linewidth=2, label="Chân ga TB (Throttle)")
    ax.plot(steps, steerings, marker="v", color="#805AD5", linewidth=2, label="Góc lái TB (Steering)")
    ax.axhline(0.40, color="orange", linestyle="--", alpha=0.7, label="Ngưỡng phạt ga (>0.40)")
    ax.set_title("Hành vi điều khiển của xe (Control Actions)")
    ax.set_xlabel("Steps / Updates")
    ax.set_ylabel("Giá trị hành động [-1, 1]")
    ax.set_ylim(-0.5, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Giá trị Value & Loss
    ax = axes[1, 1]
    has_loss = any("loss" in h and h["loss"] for h in history)
    if has_loss:
        actor_losses = [h["loss"].get("actor_loss", 0.0) if "loss" in h and h["loss"] else 0.0 for h in history]
        values = [h["loss"].get("value_mean", 0.0) if "loss" in h and h["loss"] else 0.0 for h in history]
        ax.plot(steps, actor_losses, marker=".", color="#D69E2E", label="Actor Loss")
        ax.set_ylabel("Actor Loss", color="#D69E2E")
        ax2 = ax.twinx()
        ax2.plot(steps, values, marker=".", color="#4FD1C5", linestyle=":", label="Value Estimate")
        ax2.set_ylabel("Critic Value", color="#4FD1C5")
        ax.set_title("Actor Loss & Value Estimate")
    else:
        ax.text(0.5, 0.5, "Chưa có dữ liệu Loss", ha="center", va="center")
    ax.set_xlabel("Steps / Updates")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def collect_online_steps(
    train_env: MetaDriveImageEnv,
    policy: LatentActorPolicy,
    buffer: ReplayBuffer,
    num_steps: int,
    env_state: dict,
) -> dict:
    """Thu thập num_steps bước tương tác thực tế từ môi trường MetaDrive."""
    if env_state.get("obs") is None:
        obs, _ = train_env.reset()
        policy.reset()
        env_state["obs"] = obs
        env_state["is_first"] = True
        env_state["ep_return"] = 0.0
        env_state["ep_len"] = 0

    obs = env_state["obs"]
    ep_returns = []
    successes = 0
    crashes = 0
    outs = 0

    for _ in range(num_steps):
        action = policy(obs)
        next_obs, reward, term, trunc, info = train_env.step(action)
        done = bool(term or trunc)

        buffer.add(
            Transition(
                image=obs["image"],
                state=obs["state"],
                action=np.asarray(action, dtype=np.float32).reshape(-1),
                reward=float(reward),
                done=done,
                next_image=next_obs["image"],
                next_state=next_obs["state"],
                terminated=bool(term),
            ),
            is_first=env_state["is_first"],
        )

        env_state["ep_return"] += float(reward)
        env_state["ep_len"] += 1
        env_state["is_first"] = False

        if done:
            ep_returns.append(env_state["ep_return"])
            if info.get("arrive_dest", False):
                successes += 1
            if info.get("crash", False) or info.get("crash_vehicle", False) or info.get("crash_object", False):
                crashes += 1
            if info.get("out_of_road", False):
                outs += 1

            obs, _ = train_env.reset()
            policy.reset()
            env_state["obs"] = obs
            env_state["is_first"] = True
            env_state["ep_return"] = 0.0
            env_state["ep_len"] = 0
        else:
            obs = next_obs
            env_state["obs"] = obs

    return {
        "mean_return": float(np.mean(ep_returns)) if ep_returns else 0.0,
        "completed_episodes": len(ep_returns),
        "successes": successes,
        "crashes": crashes,
        "outs": outs,
    }


def train_step_custom(
    start_states: dict[str, torch.Tensor],
    world_model: dict[str, nn.Module],
    actor_critic: ActorCritic,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    ac_cfg: dict,
    *,
    throttle_penalty_weight: float = 0.0,
    max_throttle_threshold: float = 0.4,
) -> dict[str, float]:
    """Cập nhật Actor-Critic 1 bước với tùy chọn ràng buộc chân ga."""
    horizon = int(ac_cfg.get("imagination_horizon", 15))
    gamma = float(ac_cfg.get("gamma", 0.99))
    lambda_ = float(ac_cfg.get("lambda_", 0.95))
    entropy_scale = float(ac_cfg.get("entropy_scale", 1e-4))
    grad_clip = float(ac_cfg.get("grad_clip", 100.0))

    with freeze_parameters([m for k, m in world_model.items()]):
        trajectory = imagine_rollout(
            start_states,
            world_model,
            actor_critic,
            horizon=horizon,
            gamma=gamma,
            diagnostic_control=True,
        )

        with torch.no_grad():
            from training.train_agent import lambda_return
            returns = lambda_return(
                trajectory.reward,
                trajectory.discount,
                trajectory.value[1:],
                trajectory.value[-1],
                lambda_,
            )
            weights = _continuation_weights(trajectory.discount).detach()
            weight_sum = weights.sum().clamp_min(1.0)

            ret_detach = returns.detach()
            valid_mask = weights > 0.01
            valid_returns = ret_detach[valid_mask]

            if valid_returns.numel() > 0:
                batch_p95 = torch.quantile(valid_returns, 0.95)
                batch_p05 = torch.quantile(valid_returns, 0.05)
            else:
                batch_p95 = torch.tensor(0.0, device=returns.device)
                batch_p05 = torch.tensor(0.0, device=returns.device)

            if not actor_critic.ema_initialized.item() and valid_returns.numel() > 0:
                actor_critic.ret_p95.copy_(batch_p95.detach())
                actor_critic.ret_p05.copy_(batch_p05.detach())
                actor_critic.ema_initialized.fill_(1)
            elif actor_critic.ema_initialized.item():
                actor_critic.ret_p95.mul_(0.99).add_(batch_p95.detach() * 0.01)
                actor_critic.ret_p05.mul_(0.99).add_(batch_p05.detach() * 0.01)

            scale = (actor_critic.ret_p95 - actor_critic.ret_p05).clamp_min(1.0)
            baseline = trajectory.value[:-1].detach()
            advantages = returns - baseline
            norm_advantages = advantages / scale

        # Actor Loss
        actor_objective = (weights * norm_advantages).sum() / weight_sum
        entropy = (weights * trajectory.entropy).sum() / weight_sum
        actor_loss = -actor_objective - entropy_scale * entropy

        # Ràng buộc chân ga nếu bật throttle_penalty_weight > 0
        excess_throttle_loss = torch.tensor(0.0, device=actor_loss.device)
        if throttle_penalty_weight > 0.0:
            throttle_actions = trajectory.action[:, :, 1]
            excess_th = torch.relu(throttle_actions - max_throttle_threshold).pow(2)
            excess_throttle_loss = (weights * excess_th).sum() / weight_sum
            actor_loss = actor_loss + throttle_penalty_weight * excess_throttle_loss

        actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        actor_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor_critic.actor.parameters(), grad_clip, error_if_nonfinite=True
        )
        actor_optimizer.step()

        # Critic Loss
        horizon_size, batch_size = trajectory.reward.shape
        critic_kwargs = {}
        if actor_critic.critic.control_state_dim:
            critic_kwargs['control_state'] = trajectory.control[:-1].detach().reshape(horizon_size * batch_size, -1)
        critic_value = actor_critic.critic(
            trajectory.h[:-1].detach().reshape(horizon_size * batch_size, -1),
            trajectory.z[:-1].detach().reshape(
                horizon_size * batch_size,
                *trajectory.z.shape[2:],
            ),
            **critic_kwargs,
        ).reshape(horizon_size, batch_size)
        critic_target = returns.detach()
        sym_target = torch.clamp(
            torch.sign(critic_target) * torch.log1p(torch.abs(critic_target)),
            -6.55,
            6.55,
        )
        critic_weights = weights.detach()
        critic_loss = (
            critic_weights * F.mse_loss(critic_value, sym_target, reduction="none")
        ).sum() / critic_weights.sum().clamp_min(1.0)

        critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_grad_norm = torch.nn.utils.clip_grad_norm_(
            actor_critic.critic.parameters(), grad_clip, error_if_nonfinite=True
        )
        critic_optimizer.step()
        actor_critic.update_slow_critic(tau=float(ac_cfg.get("slow_critic_update_interval", 0.02)))

    return {
        "actor_loss": float(actor_loss.item()),
        "critic_loss": float(critic_loss.item()),
        "throttle_penalty": float(excess_throttle_loss.item()),
        "imagined_reward_mean": float(trajectory.reward.mean().item()),
        "value_mean": float(trajectory.value.mean().item()),
        "lambda_return_mean": float(returns.mean().item()),
        "actor_grad_norm": float(actor_grad_norm.item()),
        "critic_grad_norm": float(critic_grad_norm.item()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/critic_warmup/continuous_seed0_refit_warmup_c19.pt")
    parser.add_argument("--buffer", default="data/replay_buffer/bootstrap_v2.npz")
    parser.add_argument("--interaction", choices=["offline", "online"], default="offline",
                        help="offline: Tưởng tượng trên Replay Buffer (nhanh) | online: Xe lăn bánh thật trong MetaDrive")
    # Cấu hình Offline Mode
    parser.add_argument("--total-updates", type=int, default=10000, help="Tổng số updates AC khi chạy offline (ví dụ 1000, 5000, 10000)")
    parser.add_argument("--eval-every", type=int, default=500, help="Đánh giá định kỳ mỗi N updates khi chạy offline")
    # Cấu hình Online Mode
    parser.add_argument("--total-steps", type=int, default=400000, help="Tổng số bước môi trường MetaDrive khi chạy online (đợt cũ: 400.000)")
    parser.add_argument("--steps-per-iter", type=int, default=1000, help="Số bước xe lăn bánh mỗi vòng trước khi train AC")
    parser.add_argument("--ac-updates-per-iter", type=int, default=25, help="Số AC updates sau mỗi vòng lăn bánh (đợt cũ: 25)")
    parser.add_argument("--eval-every-steps", type=int, default=20000, help="Đánh giá định kỳ mỗi N bước môi trường khi chạy online")
    parser.add_argument("--ckpt-every-steps", type=int, default=20000, help="Lưu checkpoint mỗi N bước môi trường khi chạy online")
    # Cấu hình Thuật toán
    parser.add_argument("--mode", choices=["raw", "safe", "constrained"], default="safe",
                        help="raw: thuần RL | safe: P3 Safe Wrapper | constrained: Safe + Phạt chân ga > 0.4")
    parser.add_argument("--throttle-penalty", type=float, default=5.0, help="Trọng số phạt chân ga khi mode=constrained")
    parser.add_argument("--no-early-stop", action="store_true", help="Chạy hết toàn bộ không ngắt giữa chừng")
    parser.add_argument("--early-stop-threshold", type=float, default=30.0, help="Ngưỡng route tối thiểu nếu có early-stop")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default="checkpoints/exp_long_test")
    parser.add_argument("--log-dir", default="logs/exp_long_test")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    log_dir = Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("      THỰC NGHIỆM HUẤN LUYỆN RL DÀI HẠN (LONG RL EXPERIMENT)")
    print("=" * 75)
    print(f"Device: {device}")
    print(f"Phương thức (--interaction): {args.interaction.upper()}")
    print(f"Chế độ thuật toán (--mode): {args.mode.upper()}")
    if args.interaction == "online":
        print(f"Tổng số bước môi trường (--total-steps): {args.total_steps:,} env steps (~{args.total_steps/1000:.0f} iterations)")
        print(f"Mỗi vòng: Thu thập {args.steps_per_iter} bước -> Cập nhật {args.ac_updates_per_iter} AC updates")
        print(f"Chu kỳ đánh giá & lưu checkpoint: mỗi {args.eval_every_steps:,} env steps")
    else:
        print(f"Tổng số updates AC (--total-updates): {args.total_updates:,} updates")
        print(f"Chu kỳ đánh giá định kỳ: mỗi {args.eval_every} updates")
    print(f"Cơ chế Early-Stop: {'TẮT (Chạy toàn bộ)' if args.no_early_stop else f'BẬT (Ngưỡng < {args.early_stop_threshold}%)'}")
    print(f"Checkpoint đầu vào: {args.checkpoint}")
    print(f"Output Checkpoints: {out_dir}")
    print(f"Output Logs: {log_dir}")
    print("=" * 75)

    # 1. Load Checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    wm_cfg = ckpt["wm_cfg"]
    ac_cfg = ckpt["actor_critic_cfg"]

    world_model = build_world_model(
        wm_cfg,
        tuple(ckpt["image_shape"]),
        ckpt["state_dim"],
        ckpt["action_dim"],
        device,
        include_decoder=False,
    )
    load_world_model_state(world_model, ckpt)

    rssm = world_model["rssm"]
    orig_reward = world_model["reward"]
    orig_continue = world_model["continue"]

    if args.mode in ["safe", "constrained"]:
        world_model["reward"] = P3SafeWrappedPredictor(orig_reward, orig_continue, rssm.continuous, "reward").to(device)
        world_model["continue"] = P3SafeWrappedPredictor(orig_reward, orig_continue, rssm.continuous, "continue").to(device)
        print("[Setup] Đã tích hợp P3 Safe Wrapper (phạt -20 điểm va chạm, cắt lệch làn > 0.28).")
    else:
        print("[Setup] Chế độ RAW: Giữ nguyên các đầu mạng dự đoán gốc của World Model.")

    actor_critic = ActorCritic(
        ac_cfg,
        deter_dim=rssm.deter_dim,
        stoch_size=rssm.stoch_size,
        action_dim=ckpt["action_dim"],
    ).to(device)
    actor_critic.load_state_dict(ckpt["actor_critic"])

    # Đóng băng World Model
    for m in world_model.values():
        m.eval().requires_grad_(False)

    actor_opt = torch.optim.Adam(actor_critic.actor.parameters(), lr=float(ac_cfg.get("actor_lr", 3e-5)))
    critic_opt = torch.optim.Adam(actor_critic.critic.parameters(), lr=float(ac_cfg.get("critic_lr", 3e-5)))

    # 2. Load Replay Buffer
    print(f"\n[Buffer] Đang tải Replay Buffer: {args.buffer}")
    buffer = ReplayBuffer.load(Path(args.buffer))
    buffer.sequence_length = 32
    print(f"[Buffer] Kích thước hiện tại: {len(buffer)} transitions.")

    eval_policy = LatentActorPolicy(world_model["encoder"], world_model["rssm"], actor_critic, device=device, deterministic=True)
    env_cfg_base = dict(ckpt["env_cfg"])

    # 3. Đánh giá Baseline tại Update 0
    print("\n--- [Khởi đầu] Đang chạy đánh giá baseline trên 8 kịch bản Paired Benchmark ---")
    m0 = evaluate_paired_benchmark(eval_policy, env_cfg_base, device)
    print(f"-> Baseline: Route = {m0['route_mean']:.2f}% (std={m0['route_std']:.2f}%), Finish = {m0['finish']}/8, Out = {m0['out']}/8, Crash = {m0['crash']}/8")
    print(f"   Hành vi: Chân ga TB = {m0['throttle_mean']:.3f}, Góc lái TB = {m0['steering_mean']:.3f}")

    metrics_log_path = log_dir / "metrics.jsonl"
    with open(metrics_log_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({"step": 0, "eval": m0, "loss": {}}) + "\n")

    history = [{"step": 0, "eval": m0}]
    plot_training_curves(history, log_dir / "training_curves.png")
    start_time = time.time()
    th_weight = args.throttle_penalty if args.mode == "constrained" else 0.0
    stopped_early = False
    stop_reason = ""

    # =========================================================================
    # NHÁNH 1: ONLINE INTERACTION (XE LĂN BÁNH THẬT TRONG METADRIVE)
    # =========================================================================
    if args.interaction == "online":
        print(f"\n[Online] Mở rộng dung lượng buffer và khởi tạo môi trường MetaDrive...")
        buffer.make_writable(max(100000, len(buffer)))
        train_env = make_env(env_cfg_base)
        env_state = {}
        total_steps = args.total_steps
        steps_per_iter = args.steps_per_iter
        ac_updates = args.ac_updates_per_iter
        current_steps = 0

        print(f"\n>>> BẮT ĐẦU CHU TRÌNH ONLINE {total_steps:,} BƯỚC...")
        while current_steps < total_steps:
            collect_info = collect_online_steps(train_env, eval_policy, buffer, steps_per_iter, env_state)
            current_steps += steps_per_iter

            last_loss = {}
            for _ in range(ac_updates):
                batch_np = buffer.sample(batch_size=8, term_ratio=0.25)
                batch = replay_batch_to_torch(batch_np, device=device)
                start_states = posterior_start_states(batch, world_model, max_states=64, deterministic=False, include_control=True)
                last_loss = train_step_custom(start_states, world_model, actor_critic, actor_opt, critic_opt, ac_cfg, throttle_penalty_weight=th_weight, max_throttle_threshold=0.4)

            print(f"Step {current_steps:6d}/{total_steps} | Return: {collect_info['mean_return']:6.2f} | Out: {collect_info['outs']} | Crash: {collect_info['crashes']} | Loss: {last_loss.get('actor_loss', 0.0):.4f} | Val: {last_loss.get('value_mean', 0.0):.2f}")

            # Đánh giá & Lưu Checkpoint định kỳ
            if current_steps % args.eval_every_steps == 0 or current_steps >= total_steps:
                print(f"\n--- [Env Step {current_steps}] Đang đánh giá đối chứng trên Paired Benchmark ---")
                train_env.close()
                m_eval = evaluate_paired_benchmark(eval_policy, env_cfg_base, device)
                train_env = make_env(env_cfg_base)
                env_state["obs"] = None

                print(f"-> Step {current_steps:6d}: Route = {m_eval['route_mean']:.2f}%, Finish = {m_eval['finish']}/8, Out = {m_eval['out']}/8, Crash = {m_eval['crash']}/8")
                print(f"   Hành vi: Chân ga TB = {m_eval['throttle_mean']:.3f}, Góc lái TB = {m_eval['steering_mean']:.3f}")

                record = {"step": current_steps, "eval": m_eval, "loss": last_loss}
                history.append(record)
                plot_training_curves(history, log_dir / "training_curves.png")
                with open(metrics_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

                if current_steps % args.ckpt_every_steps == 0 or current_steps >= total_steps:
                    ckpt_save_path = out_dir / f"checkpoint_step_{current_steps:06d}.pt"
                    torch.save({
                        "actor_critic": actor_critic.state_dict(),
                        "actor_critic_cfg": ac_cfg,
                        "wm_cfg": wm_cfg,
                        "env_cfg": ckpt["env_cfg"],
                        "env_steps": current_steps,
                        "metrics": m_eval,
                    }, ckpt_save_path)
                    print(f"   [Checkpoint] Đã lưu tại: {ckpt_save_path}")

                if not args.no_early_stop and m_eval["route_mean"] < args.early_stop_threshold:
                    stopped_early = True
                    stop_reason = f"Route completion rơi xuống {m_eval['route_mean']:.2f}% (dưới ngưỡng {args.early_stop_threshold}%)."
                    print(f"\n[ALERT - EARLY STOPPING KÍCH HOẠT]: {stop_reason}")
                    break
        train_env.close()

    # =========================================================================
    # NHÁNH 2: OFFLINE IMAGINATION (HUẤN LUYỆN SIÊU TỐC QUA TƯỞNG TƯỢNG)
    # =========================================================================
    else:
        print(f"\n>>> BẮT ĐẦU {args.total_updates:,} UPDATES HUẤN LUYỆN TƯỞNG TƯỢNG...")
        for u_idx in range(1, args.total_updates + 1):
            batch_np = buffer.sample(batch_size=8, term_ratio=0.25)
            batch = replay_batch_to_torch(batch_np, device=device)

            start_states = posterior_start_states(
                batch,
                world_model,
                max_states=64,
                deterministic=False,
                include_control=True,
            )

            loss_info = train_step_custom(
                start_states,
                world_model,
                actor_critic,
                actor_opt,
                critic_opt,
                ac_cfg,
                throttle_penalty_weight=th_weight,
                max_throttle_threshold=0.4,
            )

            if u_idx % 25 == 0 or u_idx == args.total_updates:
                print(f"Update {u_idx:5d}/{args.total_updates} | Actor Loss: {loss_info['actor_loss']:6.4f} | Critic Loss: {loss_info['critic_loss']:6.4f} | Img Rew: {loss_info['imagined_reward_mean']:6.4f} | Val: {loss_info['value_mean']:6.2f}")

            # Đánh giá định kỳ
            if u_idx % args.eval_every == 0 or u_idx == args.total_updates:
                print(f"\n--- [Update {u_idx}] Đang đánh giá đối chứng trên Paired Benchmark ---")
                m_eval = evaluate_paired_benchmark(eval_policy, env_cfg_base, device)
                print(f"-> Update {u_idx:5d}: Route = {m_eval['route_mean']:.2f}%, Finish = {m_eval['finish']}/8, Out = {m_eval['out']}/8, Crash = {m_eval['crash']}/8")
                print(f"   Hành vi: Chân ga TB = {m_eval['throttle_mean']:.3f}, Góc lái TB = {m_eval['steering_mean']:.3f}")

                record = {"update": u_idx, "eval": m_eval, "loss": loss_info}
                history.append(record)
                plot_training_curves(history, log_dir / "training_curves.png")
                with open(metrics_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

                ckpt_save_path = out_dir / f"checkpoint_update_{u_idx:05d}.pt"
                torch.save({
                    "actor_critic": actor_critic.state_dict(),
                    "actor_critic_cfg": ac_cfg,
                    "wm_cfg": wm_cfg,
                    "env_cfg": ckpt["env_cfg"],
                    "update": u_idx,
                    "metrics": m_eval,
                }, ckpt_save_path)
                print(f"   [Checkpoint] Đã lưu tại: {ckpt_save_path}")

                if not args.no_early_stop and m_eval["route_mean"] < args.early_stop_threshold:
                    stopped_early = True
                    stop_reason = f"Route completion rơi xuống {m_eval['route_mean']:.2f}% (dưới ngưỡng {args.early_stop_threshold}%)."
                    print(f"\n[ALERT - EARLY STOPPING KÍCH HOẠT]: {stop_reason}")
                    break

    elapsed = time.time() - start_time
    print("\n" + "=" * 75)
    print(f"HOÀN TẤT THỰC NGHIỆM trong {elapsed:.2f}s ({elapsed/3600:.2f} giờ).")
    print(f"Dừng sớm: {stopped_early} {f'({stop_reason})' if stopped_early else ''}")
    print(f"Toàn bộ log được ghi tại: {metrics_log_path}")
    print("=" * 75)

    summary_path = log_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "interaction": args.interaction,
        "mode": args.mode,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "elapsed_seconds": elapsed,
        "history": history,
    }, indent=2), encoding="utf-8")
    plot_training_curves(history, log_dir / "training_curves.png")
    print(f"Summary JSON: {summary_path}")
    print(f"Biểu đồ tiến trình (PNG): {log_dir / 'training_curves.png'}")


if __name__ == "__main__":
    main()
