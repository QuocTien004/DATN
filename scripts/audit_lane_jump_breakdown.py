"""Direction A Audit: Decompose Lateral & Heading error by Lane-Reference Jump vs Continuous windows.

Analyzes moving_off_bc on test and confirm splits for:
- Old prior (baseline)
- Refit exact memory seed 0
- Refit exact memory seed 1
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from models.continuous_state import ContinuousState
from models.encoder import Encoder
from scripts.train_refit_exact_memory import load_dataset
from utils.checkpoint import load_checkpoint


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / "logs" / "refit_p2_exact_memory"
    data_dir = Path("d:/DoAnTotNghiep/worldmodel-metadrive/data/offbc_episodes")
    index_path = Path("d:/DoAnTotNghiep/worldmodel-metadrive/logs/p2_eval/offbc_dataset_index.json")

    print(f"=== DIRECTION A: LANE REFERENCE & GEOMETRY AUDIT (Device: {device}) ===")

    # 1. Load baseline & branches
    ck = load_checkpoint("d:/DoAnTotNghiep/worldmodel-metadrive/checkpoints/p0_baseline/continuous_seed0_bc2000.pt", map_location="cpu")
    enc = Encoder(ck["wm_cfg"], 19, tuple(ck.get("image_shape", (64, 64, 3))))
    rssm_sd = ck["models"]["rssm"]
    cont_sd = {k[len("continuous.") :]: v for k, v in rssm_sd.items() if k.startswith("continuous.")}

    # Old prior
    cfg_old = copy.deepcopy(ck["wm_cfg"]["rssm"]["continuous_state"])
    cfg_old["exact_action_memory"] = False
    old_branch = ContinuousState(cfg_old, embed_dim=enc.embed_dim, action_dim=2).to(device)
    old_branch.load_state_dict(cont_sd, strict=True)
    old_branch.eval()

    # Exact seed 0
    cfg_exact = copy.deepcopy(ck["wm_cfg"]["rssm"]["continuous_state"])
    cfg_exact["exact_action_memory"] = True
    exact_s0 = ContinuousState(cfg_exact, embed_dim=enc.embed_dim, action_dim=2).to(device)
    exact_s0.load_state_dict(torch.load(out_dir / "refit_exact_seed0_2000.pt", map_location=device)["branch"], strict=True)
    exact_s0.eval()

    # Exact seed 1
    exact_s1 = ContinuousState(cfg_exact, embed_dim=enc.embed_dim, action_dim=2).to(device)
    exact_s1.load_state_dict(torch.load(out_dir / "refit_exact_seed1_2000.pt", map_location=device)["branch"], strict=True)
    exact_s1.eval()

    branches = {
        "Old Prior (Baseline)": old_branch,
        "Refit Exact Seed 0": exact_s0,
        "Refit Exact Seed 1": exact_s1,
    }

    # 2. Load dataset
    episodes, _ = load_dataset(index_path, data_dir)

    jump_threshold = 0.20
    audit_results = {}

    with torch.no_grad():
        for split in ["test", "confirm"]:
            audit_results[split] = {}
            print(f"\n--- Analyzing Split: {split.upper()} ---")

            for b_name, b_model in branches.items():
                audit_results[split][b_name] = {}
                for h in [1, 5, 15]:
                    pred_list, gt_list, hold_list = [], [], []
                    jump_win_list, stuck_win_list = [], []

                    for e in episodes[split]:
                        if e["meta"]["mode"] not in ["noise", "swerve", "fast"]:
                            continue
                        n = e["n"]
                        if n < h:
                            continue

                        c_seq = torch.as_tensor(e["c"], device=device)
                        act_seq = torch.as_tensor(e["action"], device=device)
                        st_np = e["state"]

                        # Detect jump steps in state[t, 8]
                        lat = st_np[:, 8]
                        lat_delta = np.abs(lat[1:] - lat[:-1])
                        jump_steps = lat_delta > jump_threshold

                        win_jumps = np.array([bool(jump_steps[t : t + h].any()) for t in range(n - h + 1)], dtype=bool)
                        is_stuck_ep = (e["meta"]["name"] == "test_44205_swerve")
                        win_stuck = np.full(n - h + 1, is_stuck_ep, dtype=bool)

                        idx = torch.arange(n - h + 1, device=device)
                        hold_phys = b_model.decode(c_seq[:-h]).cpu().numpy()

                        preds = []
                        for ix in idx.split(256):
                            act_win = act_seq[ix[None, :] + torch.arange(h, device=device)[:, None]]
                            curr = c_seq[ix]
                            for step_act in act_win.unbind(dim=0):
                                curr = b_model.imagine(curr, step_act)
                            preds.append(b_model.decode(curr).cpu().numpy())

                        pred_list.append(np.concatenate(preds, axis=0))
                        gt_list.append(st_np[h:])
                        hold_list.append(hold_phys)
                        jump_win_list.append(win_jumps)
                        stuck_win_list.append(win_stuck)

                    p_all = np.concatenate(pred_list, axis=0).astype(np.float64)
                    g_all = np.concatenate(gt_list, axis=0).astype(np.float64)
                    h_all = np.concatenate(hold_list, axis=0).astype(np.float64)
                    j_all = np.concatenate(jump_win_list, axis=0)
                    s_all = np.concatenate(stuck_win_list, axis=0)

                    # Filters
                    filters = {
                        "all_windows": np.ones(len(p_all), dtype=bool),
                        "continuous_no_stuck": (~j_all) & (~s_all),
                        "jump_only": j_all,
                        "stuck_only": s_all,
                    }

                    h_dict = {}
                    for f_name, mask in filters.items():
                        if not mask.any():
                            continue
                        mse_p = ((p_all[mask] - g_all[mask]) ** 2).mean(axis=0)
                        mse_h = ((h_all[mask] - g_all[mask]) ** 2).mean(axis=0)
                        h_ratio = mse_p / np.maximum(mse_h, 1e-12)

                        h_dict[f_name] = {
                            "n": int(mask.sum()),
                            "heading_hold_ratio": float(h_ratio[2]),
                            "lateral_hold_ratio": float(h_ratio[8]),
                            "lateral_rmse": float(np.sqrt(mse_p[8])),
                        }
                    audit_results[split][b_name][h] = h_dict

    # Save to json
    (out_dir / "lane_jump_breakdown.json").write_text(json.dumps(audit_results, indent=2), encoding="utf-8")

    # Print Clean Summary Table for Test Split
    print("\n" + "=" * 105)
    print("BẢNG BÓC TÁCH CHI TIẾT TẬP TEST (moving_off_bc): CỬA SỔ LIÊN TỤC VS CỬA SỔ NHẢY LÀN / KẸT")
    print("=" * 105)
    print(f"{'Mô hình':28s} | {'H':2s} | {'Loại cửa sổ':22s} | {'Số lượng':8s} | {'Heading Hold':12s} | {'Lateral Hold':12s} | {'Gate < 1?'}")
    print("-" * 105)

    for h in [1, 5, 15]:
        for b_name in branches:
            for f_name in ["all_windows", "continuous_no_stuck", "jump_only", "stuck_only"]:
                d = audit_results["test"][b_name][h].get(f_name)
                if d:
                    lat = d["lateral_hold_ratio"]
                    status = "PASS" if lat < 1.0 else "FAIL"
                    f_display = {
                        "all_windows": "1. Tất cả cửa sổ (All)",
                        "continuous_no_stuck": "2. Liên tục (No Jump)",
                        "jump_only": "3. Nhảy làn (Jump Only)",
                        "stuck_only": "4. Episode kẹt (44205)",
                    }[f_name]
                    print(f"{b_name:28s} | {h:2d} | {f_display:22s} | {d['n']:8d} | {d['heading_hold_ratio']:12.3f} | {lat:12.3f} | {status:8s}")
        print("-" * 105)


if __name__ == "__main__":
    main()
