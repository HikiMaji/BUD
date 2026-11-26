# -*- coding: utf-8 -*-
"""
步骤 ⑤-1：在 FIT-11 连续序列上训练修复两处真 Bug 后的 HEAL 64ch HetShield ConvGRU 并标定门限。

严格规范：
1. 仅使用 FIT-11 父序列（绝不访问 DEV-8 / OPV2V_VAL / opv2v_test）；
2. 【Bug 1 修复】严格使用 z_{t-1} -> z_t 一步时序预测损失 MSE(pred_t, z_t)，杜绝 z_cur 自重构；
3. 【Bug 2 修复】每条父序列独立复位 ConvGRU 隐状态，杜绝跨序列隐状态泄露；
4. 同时训练：
   - Shared ConvGRU（全局共享预测器，用于原论文参数版与 Fit-Tuned 版）；
   - PerSensor ConvGRU（m1 / m2 各自专有预测器，用于 HetShield-PerSensor 版）；
5. 训练完成后在 FIT-11 留出序列尾部帧上验证一步预测 MSE 收敛，并输出：
   - 原论文参数版（alpha=0.5, tau_th=0.3, T=0.1）
   - Fit-Tuned 全局 95% 分位门限
   - Fit-PerSensor 分模态 95% 分位门限。
"""

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

PROJ_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust"
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.hetshield_heal import HetShieldHEALDefender, TemporalConvGRUPredictorHEAL
from src_v2_mm.step4_1_v2_core import FIT_SEQ_NAMES, PROTO_H, PROTO_W, TRAIN_FULL19_DIR

DEFAULT_SPLIT = os.path.join(PROJ_ROOT, "results_rq1_v2_mm", "step4_1_v2_split_plan.json")
DEFAULT_CKPT_DIR = os.path.join(PROJ_ROOT, "checkpoints_v2_mm")
DEFAULT_OUT_CALIB = os.path.join(PROJ_ROOT, "results_v3", "a2_hetshield_calib.json")


def _extract_warped_pair(wrapper, ds, idx: int):
    """提取单帧对齐至本车 ego 坐标系的 (feat_ego, feat_cav_warped, valid_ego)（全部 1x64x256x256）。"""
    from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
    from opencood.tools import train_utils

    batch = train_utils.to_device(ds.collate_batch_test([ds[idx]]), wrapper.device_obj)
    p = wrapper.extract_protocol_features(batch["ego"])
    feat = p["heter_feature_2d"]
    tmat01 = p["affine_matrix"][0, 0, 1:2]
    ac = wrapper.model.pyramid_backbone.align_corners
    f_ego = feat[0:1].detach()
    f_cav = warp_affine_simple(feat[1:2], tmat01, (PROTO_H, PROTO_W), align_corners=ac).detach()
    v_ego = warp_affine_simple(p["cam_valid_mask"][1:2].float(), tmat01, (PROTO_H, PROTO_W), align_corners=ac).detach()
    return f_ego, f_cav, (v_ego > 0.5).float()


def _dist_summary(vals: list) -> dict:
    arr = np.asarray(vals, dtype=np.float64)
    if arr.size == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "n": int(arr.size),
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std()), 6),
        "p50": round(float(np.percentile(arr, 50.0)), 6),
        "p90": round(float(np.percentile(arr, 90.0)), 6),
        "p95": round(float(np.percentile(arr, 95.0)), 6),
        "p99": round(float(np.percentile(arr, 99.0)), 6),
        "max": round(float(arr.max()), 6),
    }


def calibrate_hetshield_single_step_protocol(
    def_shared: HetShieldHEALDefender,
    def_per: HetShieldHEALDefender,
    cached_seqs: OrderedDict,
    fit_seqs: list,
    stride: int = 1,
    tail_ratio: float = 0.85,
) -> dict:
    """
    A2 标定与评测严格同协议：对每一帧先 reset_sequence，用前一帧以 update_state=True 预热一步，
    再在当前帧传入真实帧号 idx_v，以 update_state=False 打分。
    """
    comp_shared = {m: {"tau_k": [], "sigma_k": [], "s_k": [], "g_k": []} for m in ("m1", "m2")}
    comp_per = {m: {"tau_k": [], "sigma_k": [], "s_k": [], "g_k": []} for m in ("m1", "m2")}

    dev = next(def_shared.parameters()).device
    with torch.no_grad():
        for s in fit_seqs:
            for mod in ("m1", "m2"):
                frames = cached_seqs[s][mod]
                n_tr = max(2, int(len(frames) * float(tail_ratio)))
                cid = f"{s}_{mod}_cav1"
                for pos in range(n_tr, len(frames)):
                    idx_v, e_h, c_h, v_b = frames[pos]
                    def_shared.reset_sequence(s)
                    def_per.reset_sequence(s)

                    if pos > 0:
                        idx_prev, pe_h, pc_h, pv_b = frames[pos - 1]
                        if int(idx_v) - int(idx_prev) == int(stride):
                            pfe = pe_h.to(dev).float()
                            pfc = pc_h.to(dev).float()
                            pve = pv_b.to(dev).float()
                            p_items = [{"cav_id": cid, "mod": mod, "feat_warped": pfc, "valid_ego": pve}]
                            def_shared.step_frame(s, int(idx_prev), pfe, p_items, update_state=True)
                            def_per.step_frame(s, int(idx_prev), pfe, p_items, update_state=True)

                    fe = e_h.to(dev).float()
                    fc = c_h.to(dev).float()
                    ve = v_b.to(dev).float()
                    c_items = [{"cav_id": cid, "mod": mod, "feat_warped": fc, "valid_ego": ve}]
                    rs = def_shared.step_frame(s, int(idx_v), fe, c_items, update_state=False)[0]
                    rp = def_per.step_frame(s, int(idx_v), fe, c_items, update_state=False)[0]

                    for k in ("tau_k", "sigma_k", "s_k", "g_k"):
                        comp_shared[mod][k].append(float(rs[k]))
                        comp_per[mod][k].append(float(rp[k]))

    return {"shared": comp_shared, "persensor": comp_per}


def main():
    parser = argparse.ArgumentParser(description="A2：训练/重标定原论文公式复现版 HEAL 64ch HetShield ConvGRU")
    parser.add_argument("--data_dir", type=str, default=TRAIN_FULL19_DIR)
    parser.add_argument("--split_json", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_steps_per_seq", type=int, default=70)
    parser.add_argument("--alpha", type=float, default=0.6, help="原论文默认 alpha=0.6")
    parser.add_argument("--energy_thresh", type=float, default=0.5)
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--out_calib", type=str, default=DEFAULT_OUT_CALIB)
    parser.add_argument("--recalib_only", action="store_true", help="加载已有 ConvGRU 权重直接重新标定")
    args = parser.parse_args()

    t0 = time.time()
    torch.manual_seed(2026)
    np.random.seed(2026)

    with open(args.split_json, "r", encoding="utf-8") as f:
        sp = json.load(f)
    fit_seqs = list(sp["fit_sequences"].keys())
    eval_seqs = set(sp["eval_sequences"].keys())
    assert set(fit_seqs).isdisjoint(eval_seqs), "Fit 与 Eval 序列存在重叠！"

    wrapper = HEALBaseWrapper(device="cuda")
    ds_m1 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m1")
    ds_m2 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m2")
    sample_seqs = wrapper.get_sample_sequences(ds_m2)

    seq_to_indices = OrderedDict()
    for i, s in enumerate(sample_seqs):
        if s in fit_seqs:
            seq_to_indices.setdefault(s, []).append(i)

    # 在内存缓存 FIT-11 子采样连续序列特征（fp16 存 CPU，仅 ~1.5GB 内存，极大加速多 epoch 训练）
    print(f"[Step 5-1] 抽取 FIT-11 连续序列协议特征 (stride={args.stride}, max_steps={args.max_steps_per_seq})...", flush=True)
    cached_seqs = OrderedDict()
    total_pairs = 0
    for s in fit_seqs:
        full_idxs = seq_to_indices[s]
        picked = full_idxs[:: max(1, int(args.stride))][: int(args.max_steps_per_seq)]
        m1_seq, m2_seq = [], []
        for idx in picked:
            e1, c1, v1 = _extract_warped_pair(wrapper, ds_m1, idx)
            _, c2, v2 = _extract_warped_pair(wrapper, ds_m2, idx)
            m1_seq.append((idx, e1.half().cpu(), c1.half().cpu(), v1.bool().cpu()))
            m2_seq.append((idx, e1.half().cpu(), c2.half().cpu(), v2.bool().cpu()))
        cached_seqs[s] = {"m1": m1_seq, "m2": m2_seq}
        total_pairs += max(0, len(picked) - 1)
        print(f"  序列 {s}: {len(picked)} 帧", flush=True)

    shared_ckpt = os.path.join(args.ckpt_dir, "hetshield_heal_64ch_shared.pth")
    persensor_ckpt = os.path.join(args.ckpt_dir, "hetshield_heal_64ch_persensor.pth")
    shared_net = TemporalConvGRUPredictorHEAL(64, 64, downsample=2).cuda()
    per_nets = {
        "m1": TemporalConvGRUPredictorHEAL(64, 64, downsample=2).cuda(),
        "m2": TemporalConvGRUPredictorHEAL(64, 64, downsample=2).cuda(),
    }

    train_hist = []
    val_summary = {}
    if args.recalib_only:
        shared_net.load_state_dict(torch.load(shared_ckpt, map_location="cpu"))
        per_sd = torch.load(persensor_ckpt, map_location="cpu")
        for m in ("m1", "m2"):
            per_nets[m].load_state_dict(per_sd[m])
        shared_net.eval()
        for m in ("m1", "m2"):
            per_nets[m].eval()
    else:
        opt_s = torch.optim.Adam(shared_net.parameters(), lr=args.lr)
        opt_p = {m: torch.optim.Adam(per_nets[m].parameters(), lr=args.lr) for m in ("m1", "m2")}

        for ep in range(int(args.epochs)):
            shared_net.train()
            for m in ("m1", "m2"):
                per_nets[m].train()
            ls_all, lp_all = [], {"m1": [], "m2": []}

            for s in fit_seqs:
                for mod in ("m1", "m2"):
                    frames = cached_seqs[s][mod]
                    n_tr = max(2, int(len(frames) * 0.85))
                    h_s, h_p = None, None
                    for t in range(n_tr - 1):
                        idx_prev, _, z_prev_h, _ = frames[t]
                        idx_next, _, z_next_h, v_next_b = frames[t + 1]
                        if int(idx_next) - int(idx_prev) != int(args.stride):
                            h_s, h_p = None, None
                            continue
                        z_prev = z_prev_h.cuda().float()
                        z_next = z_next_h.cuda().float()
                        v_next = v_next_b.cuda().float()
                        denom = torch.clamp(v_next.sum() * float(z_next.shape[1]), min=1.0)

                        pred_s, h_s_new = shared_net(z_prev, h_s)
                        loss_s = (((pred_s - z_next) ** 2) * v_next).sum() / denom
                        opt_s.zero_grad()
                        loss_s.backward()
                        torch.nn.utils.clip_grad_norm_(shared_net.parameters(), 2.0)
                        opt_s.step()
                        h_s = h_s_new.detach()
                        ls_all.append(float(loss_s.item()))

                        pred_p, h_p_new = per_nets[mod](z_prev, h_p)
                        loss_p = (((pred_p - z_next) ** 2) * v_next).sum() / denom
                        opt_p[mod].zero_grad()
                        loss_p.backward()
                        torch.nn.utils.clip_grad_norm_(per_nets[mod].parameters(), 2.0)
                        opt_p[mod].step()
                        h_p = h_p_new.detach()
                        lp_all[mod].append(float(loss_p.item()))

            ep_rec = {
                "epoch": ep + 1,
                "shared_mse": round(float(np.mean(ls_all)), 6),
                "m1_mse": round(float(np.mean(lp_all["m1"])), 6),
                "m2_mse": round(float(np.mean(lp_all["m2"])), 6),
            }
            train_hist.append(ep_rec)
            print(f"  [Epoch {ep + 1}/{args.epochs}] {ep_rec}", flush=True)

        shared_net.eval()
        for m in ("m1", "m2"):
            per_nets[m].eval()
        val_mse = {"shared": [], "m1": [], "m2": [], "identity_baseline": []}
        with torch.no_grad():
            for s in fit_seqs:
                for mod in ("m1", "m2"):
                    frames = cached_seqs[s][mod]
                    n_tr = max(2, int(len(frames) * 0.85))
                    h_s, h_p = None, None
                    for t in range(len(frames) - 1):
                        z_prev = frames[t][2].cuda().float()
                        z_next = frames[t + 1][2].cuda().float()
                        v_next = frames[t + 1][3].cuda().float()
                        denom = torch.clamp(v_next.sum() * float(z_next.shape[1]), min=1.0)
                        pred_s, h_s = shared_net(z_prev, h_s)
                        pred_p, h_p = per_nets[mod](z_prev, h_p)
                        if t >= n_tr - 1:
                            val_mse["shared"].append(float((((pred_s - z_next) ** 2) * v_next).sum().item() / denom.item()))
                            val_mse[mod].append(float((((pred_p - z_next) ** 2) * v_next).sum().item() / denom.item()))
                            val_mse["identity_baseline"].append(float((((z_prev - z_next) ** 2) * v_next).sum().item() / denom.item()))

        val_summary = {k: round(float(np.mean(v)), 6) for k, v in val_mse.items() if v}
        os.makedirs(args.ckpt_dir, exist_ok=True)
        torch.save(shared_net.state_dict(), shared_ckpt)
        torch.save({m: per_nets[m].state_dict() for m in ("m1", "m2")}, persensor_ckpt)

    # A2：原论文 alpha=0.6, tau_gate=0.3, T=0.1，标定与评测严格统一为单步预热协议 + 真实 idx_v
    def_shared = HetShieldHEALDefender(
        mode="shared", alpha=float(args.alpha), tau_th=0.3, temperature=0.1, energy_thresh=args.energy_thresh
    ).cuda().eval()
    def_shared.predictor.load_state_dict(shared_net.state_dict())
    def_per = HetShieldHEALDefender(
        mode="persensor", alpha=float(args.alpha), tau_th=0.3, temperature=0.1, energy_thresh=args.energy_thresh
    ).cuda().eval()
    for m in ("m1", "m2"):
        def_per.predictors[m].load_state_dict(per_nets[m].state_dict())

    calib_raw = calibrate_hetshield_single_step_protocol(
        def_shared, def_per, cached_seqs, fit_seqs, stride=int(args.stride), tail_ratio=0.85
    )
    sk_shared = {m: calib_raw["shared"][m]["s_k"] for m in ("m1", "m2")}
    sk_per = {m: calib_raw["persensor"][m]["s_k"] for m in ("m1", "m2")}
    gk_shared = {m: np.asarray(calib_raw["shared"][m]["g_k"], dtype=np.float64) for m in ("m1", "m2")}

    all_sk_shared = sk_shared["m1"] + sk_shared["m2"]
    calib = OrderedDict({
        "meta": {
            "description": "A2：按 HetShield 原论文公式复现（alpha=0.6，特征向本车混合，单步预热同协议标定）",
            "fit_sequences": fit_seqs,
            "stride": int(args.stride),
            "epochs": int(args.epochs),
            "alpha": float(args.alpha),
            "energy_thresh": float(args.energy_thresh),
            "shared_ckpt": shared_ckpt,
            "persensor_ckpt": persensor_ckpt,
            "elapsed_sec": round(time.time() - t0, 2),
        },
        "train_mse_history": train_hist,
        "fit_tail_one_step_mse": val_summary,
        "paper_default_params": {
            "alpha": float(args.alpha),
            "tau_th": 0.3,
            "temperature": 0.1,
            "mitigation": "z_tilde_k = g_k * z_k + (1 - g_k) * z_0",
        },
        "fit_tuned_shared_tau_p95": round(float(np.percentile(all_sk_shared, 95.0)), 6),
        "fit_tuned_shared_tau_p98": round(float(np.percentile(all_sk_shared, 98.0)), 6),
        "fit_persensor_tau_p95": {
            "m1": round(float(np.percentile(sk_per["m1"], 95.0)), 6),
            "m2": round(float(np.percentile(sk_per["m2"], 95.0)), 6),
        },
        "fit_honest_gk_lt_05_rate_paper_default": {
            "m1": round(float((gk_shared["m1"] < 0.5).mean()), 6) if gk_shared["m1"].size else 0.0,
            "m2": round(float((gk_shared["m2"] < 0.5).mean()), 6) if gk_shared["m2"].size else 0.0,
        },
        "fit_clean_component_distributions": {
            mode_k: {
                m: {k: _dist_summary(calib_raw[mode_k][m][k]) for k in ("tau_k", "sigma_k", "s_k", "g_k")}
                for m in ("m1", "m2")
            }
            for mode_k in ("shared", "persensor")
        },
        "fit_clean_sk_quantiles": {
            "shared_m1": {k: round(float(np.percentile(sk_shared["m1"], q)), 6) for k, q in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))},
            "shared_m2": {k: round(float(np.percentile(sk_shared["m2"], q)), 6) for k, q in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))},
            "persensor_m1": {k: round(float(np.percentile(sk_per["m1"], q)), 6) for k, q in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))},
            "persensor_m2": {k: round(float(np.percentile(sk_per["m2"], q)), 6) for k, q in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))},
        },
    })
    os.makedirs(os.path.dirname(args.out_calib), exist_ok=True)
    with open(args.out_calib, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)
    print(f"\n[完成] HetShield 权重与 Fit 标定已保存至：{args.out_calib} (总耗时 {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
