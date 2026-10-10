"""P4: Actor-Critic Imagination Diagnostic.

Verifies that the Critic assigns lower value to bad actions (swerve) compared to BC actions,
using the P1+P2+P3 integrated World Model.
"""
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.world_model import build_world_model, load_world_model_state
from models.actor_critic import ActorCritic
from models.predictors import ModularBoundaryHead, RewardPredictor, ContinuePredictor
from training.train_agent import imagine_rollout, posterior_start_states
from utils.checkpoint import load_checkpoint

MODES = ["clean", "noise", "swerve", "fast", "timeout"]

def construct_mlp_head(ck, with_c: bool, device, D, Z) -> torch.nn.ModuleDict:
    cfg = copy.deepcopy(ck["wm_cfg"])
    cfg.setdefault("predictors", {})["continuous_state_dim"] = 19 if with_c else 0
    heads = torch.nn.ModuleDict({"reward": RewardPredictor(cfg, D, Z), "continue": ContinuePredictor(cfg, D, Z)}).to(device)
    for name, head in heads.items():
        state = copy.deepcopy(ck["models"][name])
        if with_c:
            w = state["net.0.weight"]
            state["net.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], 19)], dim=1)
        head.load_state_dict(state)
    return heads

def main():
    parser = argparse.ArgumentParser(description="P4: Actor-Critic Imagination Test")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    p4_dir = ROOT / "logs" / "p4_eval"
    p4_dir.mkdir(parents=True, exist_ok=True)
    p3_dir = ROOT / "logs" / "p3_eval"
    p2_dir = ROOT / "logs" / "p2_eval"
    data_dir = ROOT / "data" / "offbc_episodes"
    
    print(f"=== P4 Actor-Critic Imagination Test | device={device} ===")

    # 1. Load baseline checkpoint
    ckpt_dir = ROOT / "checkpoints" / "p0_baseline"
    ck = load_checkpoint(ckpt_dir / "continuous_seed0_bc2000.pt", map_location="cpu")
    wm = build_world_model(ck["wm_cfg"], tuple(ck["image_shape"]), 19, 2, device, include_decoder=False)
    load_world_model_state(wm, ck)
    
    D = wm["rssm"].deter_dim
    Z = wm["rssm"].stoch_size
    DIM = D + Z

    # Apply P2 refit
    refit_ck = torch.load(p2_dir / "refit_seed0_2000.pt", map_location=device, weights_only=False)
    wm["rssm"].continuous.load_state_dict(refit_ck["branch"])
    
    # Fast retrain of P3 ModularBoundaryHead to use it
    print("[Step 1/3] Re-training P3 ModularBoundaryHead base reward...")
    index_path = p2_dir / "offbc_dataset_index.json"
    all_meta = json.loads(index_path.read_text(encoding="utf-8"))["episodes"]
    all_meta.sort(key=lambda e: (e["split"], int(e["map"]), MODES.index(e["mode"])))
    
    train_parts = {k: [] for k in ["hard", "c_gt", "reward", "terminal"]}
    for ep in [m for m in all_meta if m["split"] == "train"]:
        with np.load(data_dir / ep["filename"]) as f:
            n = f["action"].shape[0]
            st = f["state"].astype(np.float32)
            c_gt = wm["rssm"].continuous.normalize(torch.as_tensor(st, device=device)).cpu().numpy()
            train_parts["hard"].append(f["hard"].astype(np.float32))
            train_parts["c_gt"].append(c_gt)
            for k in ["reward", "terminal"]:
                a = f[k]
                train_parts[k].append(np.concatenate([a, np.zeros_like(a[:1])]))
                
    hard_arr = np.concatenate(train_parts["hard"])
    c_gt_arr = np.concatenate(train_parts["c_gt"])
    rew_arr = np.concatenate(train_parts["reward"])
    term_arr = np.concatenate(train_parts["terminal"])
    
    feat_gt_train = torch.tensor(np.concatenate([hard_arr[1:], c_gt_arr[1:]], axis=1), device=device)
    rew_train = torch.tensor(rew_arr[:-1], device=device, dtype=torch.float32)
    term_train = torch.tensor(term_arr[:-1], device=device, dtype=torch.bool)
    y_train = rew_train.sign() * torch.log1p(rew_train.abs())
    
    old_hz = construct_mlp_head(ck, False, device, D, Z)
    mod_head = ModularBoundaryHead(
        old_hz["reward"], old_hz["reward"], old_hz["continue"], wm["rssm"].continuous.state_mean, wm["rssm"].continuous.state_std
    ).to(device)
    
    with torch.no_grad():
        clear_train, s3_train = mod_head._corner_clearance_and_speed(feat_gt_train[:, DIM:])
        moving_nonterm = (~term_train) & (s3_train > 0.05)
        if moving_nonterm.any():
            q_train = float(torch.quantile(clear_train[moving_nonterm], 0.018).item())
        else:
            q_train = 0.0725 # fallback from P3
        mod_head.edge_threshold.copy_(torch.tensor(q_train, device=device))
        
    # Replace wm heads with modular head wrappers so imagine_rollout can use them
    class HeadWrapper(torch.nn.Module):
        def __init__(self, mod_head, key):
            super().__init__()
            self.mod_head = mod_head
            self.key = key
            self.continuous_state_dim = 19
        def forward(self, h, z, c=None):
            return self.mod_head(h, z, c=c)[self.key]
            
    wm["reward"] = HeadWrapper(mod_head, "reward")
    wm["continue"] = HeadWrapper(mod_head, "continue")
    
    # Load Actor Critic
    ac = ActorCritic(ck["actor_critic_cfg"], D, Z, 2).to(device)
    ac.load_state_dict(ck["actor_critic"])
    
    for m in wm.values():
        m.eval().requires_grad_(False)
    ac.eval().requires_grad_(False)
    
    # Step 2: Extract Clean vs Swerve states from test
    print("\n[Step 2/3] Extracting Test states for imagination...")
    test_eps = [m for m in all_meta if m["split"] == "test"]
    clean_eps = [e for e in test_eps if e["mode"] == "clean"]
    swerve_eps = [e for e in test_eps if e["mode"] == "swerve"]
    
    def get_starting_states(ep_list, max_n=50):
        c_gt_list, h_list, z_list = [], [], []
        for ep in ep_list[:5]:
            with np.load(data_dir / ep["filename"]) as f:
                st = torch.tensor(f["state"].astype(np.float32), device=device)
                hard = torch.tensor(f["hard"].astype(np.float32), device=device)
                speed = st[:, 3]
                mask = speed > 0.1
                if not mask.any(): continue
                st = st[mask]
                hard = hard[mask]
                c_gt = wm["rssm"].continuous.normalize(st)
                h = hard[:, :D]
                z = hard[:, D:].reshape(-1, wm["rssm"].stoch_dim, wm["rssm"].stoch_classes)
                c_gt_list.append(c_gt[:-1])
                h_list.append(h[:-1])
                z_list.append(z[:-1])
        if not c_gt_list: return None
        c_gt = torch.cat(c_gt_list, dim=0)[:max_n]
        h = torch.cat(h_list, dim=0)[:max_n]
        z = torch.cat(z_list, dim=0)[:max_n]
        return {"c": c_gt, "h": h, "z": z}
        
    start_clean = get_starting_states(clean_eps, 100)
    start_swerve = get_starting_states(swerve_eps, 100)
    
    # Step 3: Run Imagination and compute Value
    print("\n[Step 3/3] Running Actor-Critic Imagination Test...")
    
    with torch.no_grad():
        # 1. Rollout with pure BC Actor
        traj_clean = imagine_rollout(start_clean, wm, ac, horizon=15, gamma=0.997, diagnostic_control=True)
        disc_clean = torch.cat([torch.ones_like(traj_clean.discount[:1]), traj_clean.discount[:-1]], dim=0).cumprod(dim=0)
        r_clean = (traj_clean.reward * disc_clean).sum(dim=0).mean().item()
        
        # 2. Rollout with Swerve injected at the very first step
        class SwerveActorCritic:
            def __init__(self, ac):
                self.ac = ac
                self.step = 0
            def act(self, state, deterministic=False):
                pol = self.ac.act(state, deterministic=deterministic)
                if self.step == 0:
                    pol.action[:, 0] = torch.where(torch.rand_like(pol.action[:, 0]) > 0.5, 0.8, -0.8)
                self.step += 1
                return pol
            def slow_value(self, state):
                return self.ac.slow_value(state)
            @property
            def actor(self): return self.ac.actor
                
        swerve_ac = SwerveActorCritic(ac)
        traj_swerve = imagine_rollout(start_clean, wm, swerve_ac, horizon=15, gamma=0.997, diagnostic_control=True)
        disc_swerve = torch.cat([torch.ones_like(traj_swerve.discount[:1]), traj_swerve.discount[:-1]], dim=0).cumprod(dim=0)
        r_swerve = (traj_swerve.reward * disc_swerve).sum(dim=0).mean().item()
        
    print("=" * 80)
    print("P4 IMAGINATION TEST RESULTS (Discounted Return)")
    print("=" * 80)
    print(f"BC Action     -> Imagined Return: {r_clean:7.3f} (Final Discount: {disc_clean[-1].mean().item():.2f})")
    print(f"Swerve Action -> Imagined Return: {r_swerve:7.3f} (Final Discount: {disc_swerve[-1].mean().item():.2f})")
    print("-" * 80)
    if r_swerve < r_clean:
        print("✅ SUCCESS: World Model correctly assigns lower reward return to swerve actions!")
    else:
        print("❌ FAIL: World Model did NOT penalize swerve actions.")
        
    # Write to summary
    out = {
        "r_clean": r_clean,
        "r_swerve": r_swerve,
        "success": r_swerve < r_clean
    }
    (p4_dir / "p4_diagnostic_summary.json").write_text(json.dumps(out, indent=2))
    print(f"Results saved to {p4_dir / 'p4_diagnostic_summary.json'}")

if __name__ == "__main__":
    main()
