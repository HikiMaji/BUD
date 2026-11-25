#!/usr/bin/env bash
set -eo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH=/root/autodl-tmp/proj_dir2_hetero_trust:/root/autodl-tmp/proj_dir2_hetero_trust/third_party/OpenCOOD
cd /root/autodl-tmp/proj_dir2_hetero_trust

echo "[$(date)] Re-calibrating frozen thresholds on 240 validate frames with multi-scale max scan statistic..."
/root/miniconda3/bin/python -u -c '
import json, copy, torch
import numpy as np
from src.open_hetero_env import COLLAB_PROFILES, FrozenAnchorWrapper, load_opv2v_split_dataset, prepare_hetero_frame_bundle
from src.protocol_models import HeteroAgentBranch
from src.uncertainty_fusion import BUDFusionModule

cfg = "/root/autodl-tmp/ToolV2X_resources/cad_native_2026_10_05/AdvCollaborativePerception/models/OpenCOOD/pointpillar_attentive_fusion/config.yaml"
ckpt = "/root/autodl-tmp/ToolV2X_resources/cad_native_2026_10_05/AdvCollaborativePerception/models/OpenCOOD/pointpillar_attentive_fusion/latest.pth"
val_ds, _ = load_opv2v_split_dataset(cfg, "data_links/opv2v_val", "data_links/opv2v_train", is_train=False)
anc = FrozenAnchorWrapper(cfg, ckpt, val_ds.pre_processor).cuda()
branches = {}
for p in COLLAB_PROFILES:
    b = HeteroAgentBranch(p, bounded=True, lip_gamma=1.12).cuda().eval()
    b.load_state_dict(torch.load(f"outputs/checkpoints/{p}_bounded.pth", map_location="cuda"), strict=True)
    branches[p] = b

with open("outputs/calibrated_thresholds.json", "r", encoding="utf-8") as f:
    cal = json.load(f)

bud = BUDFusionModule(384).cuda().eval()
bud.load_state_dict(torch.load("outputs/checkpoints/bud_fusion_duid.pth", map_location="cuda"), strict=True)
bud.calib_params["dom_var"] = copy.deepcopy(cal["bud_calib_params"]["dom_var"])

q_target = 1.0 - cal["alpha_budget"]
sf_full_l, sf_no_a_l, sf_no_e_l, sf_q95_l, dcal_p99, epis_p95 = [], [], [], [], [], []

stride = max(1, len(val_ds) // 240)
val_indices = list(range(0, len(val_ds), stride))[:240]
with torch.no_grad():
    for idx in val_indices:
        bdl = prepare_hetero_frame_bundle(val_ds, idx, anc, include_teacher=False)
        if not bdl["cav_entries"]:
            continue
        z_l = [bdl["z_ego"]]
        ug_l, dm_l, p_n = [], [], []
        for c in bdl["cav_entries"]:
            zc, _ = branches[c["profile"]](c["pts_hetero_t"])
            z_l.append(zc)
            ug_l.append(c["u_geom"].unsqueeze(0))
            dm_l.append(c["dist_map"].unsqueeze(0))
            p_n.append(c["profile"])
        z_all = torch.cat(z_l, dim=0)
        s_ego = bdl["sigma_alea_ego"]
        ug_c = torch.cat(ug_l, dim=0)
        dm_c = torch.cat(dm_l, dim=0)

        dc, epc, _, _, s_full, _, _ = bud.compute_verification_maps(z_all, s_ego, ug_c, dm_c, p_n, True, True, True)
        sf_full_l.extend(s_full.cpu().tolist())
        dcal_p99.append(float(torch.quantile(dc.flatten(), 0.995).item()))
        nz = epc[epc > 0.01]
        if nz.numel() > 10:
            epis_p95.append(float(torch.quantile(nz, 0.95).item()))

        _, _, _, _, s_na, _, _ = bud.compute_verification_maps(z_all, s_ego, ug_c, dm_c, p_n, False, True, True)
        sf_no_a_l.extend(s_na.cpu().tolist())
        _, _, _, _, s_ne, _, _ = bud.compute_verification_maps(z_all, s_ego, ug_c, dm_c, p_n, True, False, True)
        sf_no_e_l.extend(s_ne.cpu().tolist())
        _, _, _, _, s_q95, _, _ = bud.compute_verification_maps(z_all, s_ego, ug_c, dm_c, p_n, True, True, False)
        sf_q95_l.extend(s_q95.cpu().tolist())

bud.calib_params["tau_frame"] = round(float(np.quantile(sf_full_l, q_target)), 5)
bud.calib_params["tau_grid"] = round(float(np.quantile(dcal_p99, 0.90)), 5)
bud.calib_params["tau_epis"] = round(float(np.quantile(epis_p95, 0.90)), 5)
cal["hetero_calibrated_95"]["ours_full"] = bud.calib_params["tau_frame"]
cal["hetero_calibrated_95"]["ours_no_alea"] = round(float(np.quantile(sf_no_a_l, q_target)), 5)
cal["hetero_calibrated_95"]["ours_no_epis"] = round(float(np.quantile(sf_no_e_l, q_target)), 5)
cal["hetero_calibrated_95"]["ours_quant95"] = round(float(np.quantile(sf_q95_l, q_target)), 5)
cal["bud_calib_params"] = bud.calib_params

with open("outputs/calibrated_thresholds.json", "w", encoding="utf-8") as f:
    json.dump(cal, f, indent=2, ensure_ascii=False)
print("Updated calibrated_thresholds.json:", json.dumps(cal, indent=2))
'

echo "[$(date)] Running FULL 150-Frame Unseen OPV2V Test Set Evaluation..."
/root/miniconda3/bin/python -u scripts/step3_evaluate_all.py --max_test_frames 150
echo "[$(date)] Generating Publication-Grade Figures (Step 4 & Step 5)..."
/root/miniconda3/bin/python -u scripts/step4_plot_figures.py
/root/miniconda3/bin/python -u scripts/step5_visualize_bev_case.py
echo "[$(date)] FINAL_150_PIPELINE_ALL_DONE"
