#!/usr/bin/env python
"""Vẽ biểu đồ huấn luyện từ file metrics.jsonl.

Cách dùng:
    python scripts/plot_training_curves.py --log-dir logs/exp_long_online_400k
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import matplotlib.pyplot as plt


def plot_metrics(metrics_file: Path, out_file: Path) -> None:
    if not metrics_file.exists():
        print(f"[Lỗi] Không tìm thấy file: {metrics_file}")
        return

    history = []
    with open(metrics_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    history.append(json.loads(line))
                except Exception:
                    pass

    if not history:
        print("[Lỗi] File log rỗng hoặc không có bản ghi hợp lệ.")
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
    plt.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"-> Đã vẽ và lưu biểu đồ tại: {out_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="logs/exp_long_online_400k", help="Thư mục chứa metrics.jsonl")
    parser.add_argument("--out", default=None, help="Đường dẫn file ảnh đầu ra (mặc định training_curves.png trong log-dir)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    metrics_path = log_dir / "metrics.jsonl"
    out_path = Path(args.out) if args.out else log_dir / "training_curves.png"

    plot_metrics(metrics_path, out_path)


if __name__ == "__main__":
    main()
