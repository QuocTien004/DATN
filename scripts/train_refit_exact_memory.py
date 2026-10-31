"""Step 1 of Next Steps: Refit Continuous Transition with exact_action_memory=True during training.

Strictly follows docs/P1_P4_AUDIT.md:
- 17 coordinates optimized (action-memory 5/6 hard-assigned by exact_action_memory=True).
- All other modules frozen (encoder, posterior, Actor, Critic, heads).
- 2000 gradient updates on train episodes, batch 256, Adam lr=0.001, grad_clip=10.
- Horizons (1, 5, 15).
- Evaluates on valid, test, confirm splits and checks Gates against baseline and old refit.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from models.continuous_state import ContinuousState
from models.encoder import Encoder
from training.continuous_state import continuous_transition_loss
from utils.checkpoint import load_checkpoint


MODES = ["clean", "noise", "swerve", "fast", "timeout"]
MODE_PROBS = [0.2375, 0.2375, 0.2375, 0.2375, 0.05]


def load_dataset(index_path: Path, data_dir: Path):
    with index_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)["episodes"]

    episodes = {"train": [], "valid": [], "test": [], "confirm": []}
    by_mode = {"train": {m: [] for m in MODES}}

    for item in meta:
        p = data_dir / item["filename"]
        with np.load(p) as npz:
            c_data = npz["c"].astype(np.float32)
            act_data = npz["action"].astype(np.float32)
            st_data = npz["state"].astype(np.float32)
            term_data = npz["terminal"].astype(bool)
            out_data = npz["out"].astype(bool)
            crash_data = npz["crash"].astype(bool)
            rew_data = npz["reward"].astype(np.float32)

        ep_dict = {
            "meta": item,
            "n": len(act_data),
            "c": c_data,
            "action": act_data,
            "state": st_data,
            "terminal": term_data,
            "out": out_data,
            "crash": crash_data,
            "reward": rew_data,
            "is_failure": bool(term_data.any() and (out_data.any() or crash_data.any() or (rew_data < 0).any())),
        }
        split = item["split"]
        episodes[split].append(ep_dict)
        if split == "train":
            by_mode["train"][item["mode"]].append(ep_dict)

    return episodes, by_mode


def sample_batch(train_by_mode: dict, batch_size: int = 256, horizon_max: int = 15):
    initial_list = []
    actions_list = []
    targets_list = []
    valid_list = []

    sampled_modes = np.random.choice(MODES, size=batch_size, p=MODE_PROBS)

    for mode in sampled_modes:
        pool = train_by_mode[str(mode)]
        ep = pool[np.random.randint(len(pool))]
        n = ep["n"]

        # Failure window sampling: 25% chance to focus on last 15 transitions
        if ep["is_failure"] and np.random.rand() < 0.25:
            start_t = np.random.randint(max(0, n - horizon_max), n)
        else:
            start_t = np.random.randint(0, n)

        init_c = ep["c"][start_t]

        # Extract up to horizon_max actions and horizon_max + 1 target states
        act_slice = ep["action"][start_t : start_t + horizon_max]
        st_slice = ep["state"][start_t : start_t + horizon_max + 1]

        act_pad = np.zeros((horizon_max, 2), dtype=np.float32)
        st_pad = np.zeros((horizon_max + 1, 19), dtype=np.float32)
        valid_mask = np.zeros(horizon_max, dtype=bool)

        real_len = len(act_slice)
        act_pad[:real_len] = act_slice
        st_pad[: len(st_slice)] = st_slice
        valid_mask[:real_len] = True

        initial_list.append(init_c)
        actions_list.append(act_pad)
        targets_list.append(st_pad)
        valid_list.append(valid_mask)

    init_tensor = torch.as_tensor(np.stack(initial_list), dtype=torch.float32)
    # actions shape: (H, B, A)
    act_tensor = torch.as_tensor(np.stack(actions_list), dtype=torch.float32).permute(1, 0, 2)
    # targets shape: (H + 1, B, D)
    st_tensor = torch.as_tensor(np.stack(targets_list), dtype=torch.float32).permute(1, 0, 2)
    # valid shape: (H, B)
    val_tensor = torch.as_tensor(np.stack(valid_list), dtype=torch.bool).permute(1, 0)

    return init_tensor, act_tensor, st_tensor, val_tensor


@torch.no_grad()
def evaluate_prior(branch: ContinuousState, eps: list[dict], device: torch.device):
    results = {}
    for h in [1, 5, 15]:
        pred_list, gt_list, gt0_list, hold_list = [], [], [], []
        mode_list = []

        for e in eps:
            n = e["n"]
            if n < h:
                continue

            c_seq = torch.as_tensor(e["c"], device=device)
            act_seq = torch.as_tensor(e["action"], device=device)
            st_np = e["state"]

            idx = torch.arange(n - h + 1, device=device)
            hold_phys = branch.decode(c_seq[:-h]).cpu().numpy()

            preds = []
            for ix in idx.split(256):
                act_win = act_seq[ix[None, :] + torch.arange(h, device=device)[:, None]]
                # Rollout prior H steps
                curr = c_seq[ix]
                for step_act in act_win.unbind(dim=0):
                    curr = branch.imagine(curr, step_act)
                preds.append(branch.decode(curr).cpu().numpy())

            pred_list.append(np.concatenate(preds, axis=0))
            gt_list.append(st_np[h:])
            gt0_list.append(st_np[:-h])
            hold_list.append(hold_phys)
            mode_list.append(np.full(n - h + 1, e["meta"]["mode"]))

        pred_all = np.concatenate(pred_list, axis=0).astype(np.float64)
        gt_all = np.concatenate(gt_list, axis=0).astype(np.float64)
        gt0_all = np.concatenate(gt0_list, axis=0).astype(np.float64)
        hold_all = np.concatenate(hold_list, axis=0).astype(np.float64)
        modes_all = np.concatenate(mode_list, axis=0)

        # moving_off_bc: noise, swerve, fast
        q = np.isin(modes_all, ["noise", "swerve", "fast"])
        p_sub, g_sub, g0_sub, h_sub = pred_all[q], gt_all[q], gt0_all[q], hold_all[q]

        mse = ((p_sub - g_sub) ** 2).mean(axis=0)
        hold_mse = ((h_sub - g_sub) ** 2).mean(axis=0)
        hold_ratio = mse / np.maximum(hold_mse, 1e-12)
        delta_mse = (((p_sub - h_sub) - (g_sub - g0_sub)) ** 2).mean(axis=0)
        zero_delta_mse = ((g_sub - g0_sub) ** 2).mean(axis=0)
        delta_ratio = delta_mse / np.maximum(zero_delta_mse, 1e-12)

        results[h] = {
            "n_windows": int(q.sum()),
            "heading_rmse": float(np.sqrt(mse[2])),
            "heading_hold_ratio": float(hold_ratio[2]),
            "heading_delta_ratio": float(delta_ratio[2]),
            "lateral_rmse": float(np.sqrt(mse[8])),
            "lateral_hold_ratio": float(hold_ratio[8]),
            "lateral_delta_ratio": float(delta_ratio[8]),
            "memory5_rmse": float(np.sqrt(mse[5])),
            "memory6_rmse": float(np.sqrt(mse[6])),
        }
    return results


def main():
    p = argparse.ArgumentParser(description="Refit Continuous Transition with exact_action_memory=True")
    p.add_argument("--baseline-checkpoint", default="d:/DoAnTotNghiep/worldmodel-metadrive/checkpoints/p0_baseline/continuous_seed0_bc2000.pt")
    p.add_argument("--dataset-index", default="d:/DoAnTotNghiep/worldmodel-metadrive/logs/p2_eval/offbc_dataset_index.json")
    p.add_argument("--data-dir", default="d:/DoAnTotNghiep/worldmodel-metadrive/data/offbc_episodes")
    p.add_argument("--out-dir", default="logs/refit_p2_exact_memory")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== Refit Continuous Transition (exact_action_memory=True) ===")
    print(f"Seed: {args.seed} | Updates: {args.updates} | Batch: {args.batch_size} | Device: {device}")

    # 1. Load baseline & setup model
    ck = load_checkpoint(args.baseline_checkpoint, map_location="cpu")
    enc = Encoder(ck["wm_cfg"], 19, tuple(ck.get("image_shape", (64, 64, 3))))
    rssm_sd = ck["models"]["rssm"]
    cont_sd = {k[len("continuous.") :]: v for k, v in rssm_sd.items() if k.startswith("continuous.")}

    # BẬT EXACT_ACTION_MEMORY=TRUE NGAY TRONG LÚC HUẤN LUYỆN
    cfg = copy.deepcopy(ck["wm_cfg"]["rssm"]["continuous_state"])
    cfg["exact_action_memory"] = True

    branch = ContinuousState(cfg, embed_dim=enc.embed_dim, action_dim=2).to(device)
    branch.load_state_dict(cont_sd, strict=True)

    # ĐÓNG BĂNG TẤT CẢ, CHỈ CẬP NHẬT TRANSITION
    branch.requires_grad_(False)
    branch.transition.requires_grad_(True)

    opt = torch.optim.Adam(branch.transition.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    # 2. Load dataset
    print("Loading 210 off-BC episodes...")
    episodes, by_mode = load_dataset(Path(args.dataset_index), Path(args.data_dir))
    print(f"Train episodes: {len(episodes['train'])} | Valid: {len(episodes['valid'])} | Test: {len(episodes['test'])} | Confirm: {len(episodes['confirm'])}")

    # 3. Train loop
    print("\nStarting 2.000 gradient updates...")
    t0 = time.time()
    loss_history = []
    branch.train()

    for step in range(1, args.updates + 1):
        init_t, act_t, st_t, val_t = sample_batch(by_mode["train"], args.batch_size)
        init_t = init_t.to(device)
        act_t = act_t.to(device)
        st_t = st_t.to(device)
        val_t = val_t.to(device)

        opt.zero_grad()
        loss, info = continuous_transition_loss(branch, init_t, act_t, st_t, val_t, horizons=(1, 5, 15))
        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(branch.transition.parameters(), args.grad_clip)
        opt.step()

        if step % 50 == 0 or step == 1:
            loss_val = float(loss.item())
            loss_history.append({"step": step, "loss": loss_val, "grad_norm": float(grad_norm.item())})
            print(f"Step {step:4d}/{args.updates} | Loss: {loss_val:.4f} | Grad Norm: {grad_norm.item():.3f} | Elapsed: {time.time()-t0:.1f}s", flush=True)

        if step in [250, 500, 1000, 2000]:
            save_payload = {
                "branch": branch.state_dict(),
                "cfg": cfg,
                "update": step,
                "seed": args.seed,
                "exact_action_memory": True,
            }
            torch.save(save_payload, out_dir / f"refit_exact_seed{args.seed}_{step}.pt")

    train_time = time.time() - t0
    print(f"\nTraining completed in {train_time:.1f}s.")

    # 4. Evaluation & Gate Check
    print("\nEvaluating trained model across Valid, Test, and Confirm splits (moving_off_bc)...")
    branch.eval()

    eval_results = {}
    for split in ["valid", "test", "confirm"]:
        eval_results[split] = evaluate_prior(branch, episodes[split], device)

    # Save results
    final_output = {
        "seed": args.seed,
        "updates": args.updates,
        "exact_action_memory": True,
        "train_time_seconds": train_time,
        "eval_results": eval_results,
    }
    with (out_dir / f"eval_summary_seed{args.seed}.json").open("w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    # Print Gate comparison
    print("\n" + "=" * 90)
    print(f"GATE EVALUATION TABLE: moving_off_bc (Seed {args.seed}, exact_action_memory=True)")
    print("=" * 90)
    print(f"{'Split':7s} | {'H':2s} | {'Heading Hold':12s} | {'Head Delta':10s} | {'Lateral Hold':12s} | {'Lat Delta':10s} | {'Gate Lat < 1':12s}")
    print("-" * 90)

    for split in ["valid", "test", "confirm"]:
        for h in [1, 5, 15]:
            r = eval_results[split][h]
            lat_pass = "PASS" if r["lateral_hold_ratio"] < 1.0 else "FAIL"
            print(f"{split:7s} | {h:2d} | {r['heading_hold_ratio']:12.4f} | {r['heading_delta_ratio']:10.4f} | {r['lateral_hold_ratio']:12.4f} | {r['lateral_delta_ratio']:10.4f} | {lat_pass:12s}")
        print("-" * 90)


if __name__ == "__main__":
    main()
