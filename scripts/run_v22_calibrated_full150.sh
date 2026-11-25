#!/usr/bin/env bash
set -eo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH=/root/autodl-tmp/proj_dir2_hetero_trust:/root/autodl-tmp/proj_dir2_hetero_trust/third_party/OpenCOOD
cd /root/autodl-tmp/proj_dir2_hetero_trust

echo "[$(date)] Calibrating dimensionless component scales & joint 95% threshold on 240 validate frames..."
/root/miniconda3/bin/python -u -c '
import json, copy, torch
import numpy as np
import torch.nn.functional as F
from src.open_hetero_env import COLLAB_PROFILES, FrozenAnchorWrapper, load_opv2v_split_dataset, prepare_hetero_frame_bundle
from src.protocol_models import HeteroAgentBranch
from src.uncertainty_fusion import BUDFusionModule, mean_topk_pooled_score, extract_net_response

cfg = "/root/autodl-tmp/ToolV2X_resources/cad_native_2026_10_05/AdvCollaborativePerception/models/OpenCOOD/pointpillar_attentive_fusion/config.yaml"
ckpt = "/root/autodl-tmp/ToolV2X_resources/cad_native_2026_10_05/AdvCollaborativePerception/models/OpenCOOD/pointpillar_attentive_fusion/latest.pth"
val_ds, _ = load_opv2v_split_dataset(cfg, "data_links/opv2v_val", "data_links/opv2v_train", is_train=False)
anc = FrozenAnchorWrapper(cfg, ckpt, val_ds.pre_processor).cuda()
cls_ref = anc.model.cls_head
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
bud.calib_params["dom_var"]["profile_a_pp64"] = 0.030

stride = max(1, len(val_ds) // 240)
val_indices = list(range(0, len(val_ds), stride))[:240]

# Pass 1: cache validation bundles and measure raw_covis, raw_sem, raw_epis 95% scales
val_cache = []
raw_c_l, raw_s_l, raw_e_l = [], [], []
dcal_p99, epis_p95 = [], []

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
        sal_all = torch.sigmoid(cls_ref(z_all)).max(dim=1, keepdim=True)[0]
        val_cache.append((z_all.cpu().half(), s_ego.cpu().half(), ug_c.cpu().half(), dm_c.cpu().half(), sal_all.cpu().half(), p_n))

        # set temporary unit scales=1.0 to extract exact raw components
        bud.calib_params["scale_covis"] = 1.0
        bud.calib_params["scale_sem"] = 1.0
        bud.calib_params["scale_epis"] = 1.0
        dc, epc, _, _, _, sc_c, sc_e = bud.compute_verification_maps(z_all, s_ego, ug_c, dm_c, p_n, cls_ref, sal_all, True, True, True)
        # compute raw_sem separately via difference
        _, _, _, _, sf_no_e, _, _ = bud.compute_verification_maps(z_all, s_ego, ug_c, dm_c, p_n, cls_ref, sal_all, True, False, True)
        # re-extract exact raw_sem
        sal_ego, sal_cavs = sal_all[0:1], sal_all[1:]
        _, cav_nn = extract_net_response(z_all[1:])
        silent_ind = (cav_nn <= bud.calib_params["act_thresh"]).float()
        far_mask = (dm_c > bud.calib_params["r_near"]).float()
        ego_clear = (s_ego < bud.calib_params["theta_clear"]).float()
        cav_close = (ug_c < 0.52).float()
        sem_map = ego_clear * F.relu(sal_cavs - sal_ego - 0.18) + 0.85 * ego_clear * cav_close * F.relu(sal_ego - sal_cavs - 0.35) * (1.0 - silent_ind * far_mask)
        r_sem = mean_topk_pooled_score(sem_map, None, k=10)

        raw_c_l.extend(sc_c.cpu().tolist())
        raw_e_l.extend(sc_e.cpu().tolist())
        raw_s_l.extend(r_sem.cpu().tolist())
        dcal_p99.append(float(torch.quantile(dc.flatten(), 0.995).item()))
        nz = epc[epc > 0.01]
        if nz.numel() > 10:
            epis_p95.append(float(torch.quantile(nz, 0.95).item()))

bud.calib_params["scale_covis"] = round(max(float(np.quantile(raw_c_l, 0.95)), 0.05), 5)
bud.calib_params["scale_sem"] = round(max(float(np.quantile(raw_s_l, 0.95)), 0.03), 5)
bud.calib_params["scale_epis"] = round(max(float(np.quantile(raw_e_l, 0.95)), 0.05), 5)
bud.calib_params["tau_grid"] = round(float(np.quantile(dcal_p99, 0.92)), 5)
bud.calib_params["tau_epis"] = round(float(np.quantile(epis_p95, 0.92)), 5)

# Pass 2: calibrate exact 95% joint threshold tau_frame for full and all ablations
q_target = 1.0 - cal["alpha_budget"]
sf_full_l, sf_na_l, sf_ne_l, sf_q95_l = [], [], [], []
with torch.no_grad():
    for z_h, se_h, ug_h, dm_h, sal_h, p_n in val_cache:
        z_all, s_ego, ug_c, dm_c, sal_all = z_h.cuda().float(), se_h.cuda().float(), ug_h.cuda().float(), dm_h.cuda().float(), sal_h.cuda().float()
        sf_full_l.extend(bud.compute_verification_maps(z_all, s_ego, ug_c, dm_c, p_n, cls_ref, sal_all, True, True, True)[4].cpu().tolist())
        sf_na_l.extend(bud.compute_verification_maps(z_all, s_ego, ug_c, dm_c, p_n, cls_ref, sal_all, False, True, True)[4].cpu().tolist())
        sf_ne_l.extend(bud.compute_verification_maps(z_all, s_ego, ug_c, dm_c, p_n, cls_ref, sal_all, True, False, True)[4].cpu().tolist())
        sf_q95_l.extend(bud.compute_verification_maps(z_all, s_ego, ug_c, dm_c, p_n, cls_ref, sal_all, True, True, False)[4].cpu().tolist())

bud.calib_params["tau_frame"] = round(float(np.quantile(sf_full_l, q_target)), 5)
cal["hetero_calibrated_95"]["ours_full"] = bud.calib_params["tau_frame"]
cal["hetero_calibrated_95"]["ours_no_alea"] = round(float(np.quantile(sf_na_l, q_target)), 5)
cal["hetero_calibrated_95"]["ours_no_epis"] = round(float(np.quantile(sf_ne_l, q_target)), 5)
cal["hetero_calibrated_95"]["ours_quant95"] = round(float(np.quantile(sf_q95_l, q_target)), 5)
cal["bud_calib_params"] = bud.calib_params

with open("outputs/calibrated_thresholds.json", "w", encoding="utf-8") as f:
    json.dump(cal, f, indent=2, ensure_ascii=False)
print("Dimensionless Calibrated Params:", json.dumps(cal, indent=2))
'

echo "[$(date)] Quick 18-frame preview check..."
/root/miniconda3/bin/python -u scripts/step3_evaluate_all.py --max_test_frames 18
echo "[$(date)] Starting FULL 150-Frame Unseen Test Evaluation..."
/root/miniconda3/bin/python -u scripts/step3_evaluate_all.py --max_test_frames 150
echo "[$(date)] Plotting Step 4 & Step 5 Figures..."
/root/miniconda3/bin/python -u scripts/step4_plot_figures.py
/root/miniconda3/bin/python -u scripts/step5_visualize_bev_case.py
echo "[$(date)] V22_FULL_PIPELINE_ALL_DONE"
