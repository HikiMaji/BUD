# -*- coding: utf-8 -*-
"""
步骤 ⑤-1（v3 升级版）：在 splits_v3.json 的 calib 标定组连续序列上训练 HEAL 64ch HetShield ConvGRU。

严格规范：
1. 仅使用 splits_v3.json 的 calib 组 26 条序列（强制断言与 select / dev / validate / test / test_culvercity 零重叠）；
2. 内存安全设计：按单条序列落盘缓存至 results_v3/_m5_hetshield_seq_cache/{seq}.pt（本车 e1 仅存 1 份），
   训练与标定循环中逐序列从磁盘流式加载、用完立即释放，CPU 内存峰值 < 3 GB（彻底消除 52.8 GB OOM）；
3. 严格执行 z_{t-1} -> z_t 一步时序预测损失 MSE(pred_t, z_t)，每条父序列独立复位隐状态；
4. 同时训练 Shared ConvGRU 与 PerSensor ConvGRU（论文默认 0.3 版固定不调参，Tuned/PerSensor 正式门限在 M7 select 组搜索）；
5. 完整兼容 --max_total_frames 2 极短序列冒烟测试（自动保护尾部标定索引不为空）。
"""

import argparse
import gc
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch

PROJ_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust"
if os.path.exists(PROJ_ROOT) and PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
LOCAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if LOCAL_ROOT not in sys.path:
    sys.path.insert(0, LOCAL_ROOT)

from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
from src_v2_mm.hetshield_heal import HetShieldHEALDefender, TemporalConvGRUPredictorHEAL
from src_v2_mm.step4_1_v2_core import (
    OPV2V_TRAIN_DIR,
    PROTO_H,
    PROTO_W,
    V3_HETSHIELD_CALIB_JSON,
    V3_RESULTS_DIR,
    V3_SPLITS_JSON,
)

DEFAULT_SPLIT = V3_SPLITS_JSON
DEFAULT_CKPT_DIR = os.path.join(PROJ_ROOT if os.path.exists(PROJ_ROOT) else LOCAL_ROOT, "checkpoints_v2_mm")
DEFAULT_OUT_CALIB = V3_HETSHIELD_CALIB_JSON
DEFAULT_SEQ_CACHE_DIR = os.path.join(V3_RESULTS_DIR, "_m5_hetshield_seq_cache")


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
    return f_ego, f_cav, (v_ego > 0.5)


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


def _load_seq_entry(seq_store, seq_name: str) -> dict:
    """支持传入磁盘 .pt 路径字典或内存序列字典（供 CPU 单元测试直接传入轻量模拟张量）。"""
    val = seq_store[seq_name]
    if isinstance(val, str):
        return torch.load(val, map_location="cpu")
    return val


def _iter_mod_frames_from_entry(entry: dict, mod: str):
    """兼容新版紧凑结构 {indices, ego, m1_cav, m1_val, m2_cav, m2_val} 与旧版 {m1: [(idx, e, c, v)], m2: ...}。"""
    if mod in entry and isinstance(entry[mod], list):
        return entry[mod]
    idxs = entry["indices"]
    egos = entry["ego"]
    cavs = entry[f"{mod}_cav"]
    vals = entry[f"{mod}_val"]
    return [(idxs[i], egos[i], cavs[i], vals[i]) for i in range(len(idxs))]


def calibrate_hetshield_single_step_protocol(
    def_shared: HetShieldHEALDefender,
    def_per: HetShieldHEALDefender,
    seq_store: dict,
    fit_seqs: list,
    stride: int = 1,
    tail_ratio: float = 0.85,
) -> dict:
    """
    A2 标定与评测严格同协议：逐序列流式加载，对每一帧先 reset_sequence，用前一帧以 update_state=True 预热一步，
    再在当前帧传入真实帧号 idx_v，以 update_state=False 打分。
    自动保护极短冒烟序列（如 2 帧），保证至少最后 1 帧进入标定统计，绝不产生空列表。
    """
    comp_shared = {m: {"tau_k": [], "sigma_k": [], "s_k": [], "g_k": []} for m in ("m1", "m2")}
    comp_per = {m: {"tau_k": [], "sigma_k": [], "s_k": [], "g_k": []} for m in ("m1", "m2")}

    dev = next(def_shared.parameters()).device
    with torch.no_grad():
        for s in fit_seqs:
            entry = _load_seq_entry(seq_store, s)
            for mod in ("m1", "m2"):
                frames = _iter_mod_frames_from_entry(entry, mod)
                if not frames:
                    continue
                n_tr = int(len(frames) * float(tail_ratio))
                if n_tr >= len(frames):
                    n_tr = max(0, len(frames) - 1)
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
            del entry

    return {"shared": comp_shared, "persensor": comp_per}


def _extract_or_load_hs_seq(
    wrapper: HEALBaseWrapper,
    ds_m1,
    ds_m2,
    seq_name: str,
    picked_indices: list,
    cache_dir: str,
    persist_to_disk: bool = True,
):
    if persist_to_disk:
        os.makedirs(cache_dir, exist_ok=True)
        pt_path = os.path.join(cache_dir, f"{seq_name}.pt")
        if os.path.exists(pt_path):
            meta_check = torch.load(pt_path, map_location="cpu")
            if meta_check.get("indices") == list(picked_indices):
                print(f"  [Step 5-1 缓存命中] 序列 {seq_name}: {len(picked_indices)} 帧", flush=True)
                del meta_check
                return pt_path

    egos, m1_cavs, m1_vals, m2_cavs, m2_vals = [], [], [], [], []
    for idx in picked_indices:
        e1, c1, v1 = _extract_warped_pair(wrapper, ds_m1, idx)
        _, c2, v2 = _extract_warped_pair(wrapper, ds_m2, idx)
        egos.append(e1.half().cpu())
        m1_cavs.append(c1.half().cpu())
        m1_vals.append(v1.bool().cpu())
        m2_cavs.append(c2.half().cpu())
        m2_vals.append(v2.bool().cpu())

    payload = {
        "seq_name": seq_name,
        "indices": list(picked_indices),
        "ego": egos,
        "m1_cav": m1_cavs,
        "m1_val": m1_vals,
        "m2_cav": m2_cavs,
        "m2_val": m2_vals,
    }
    print(f"  [Step 5-1 抽取完成] 序列 {seq_name}: {len(picked_indices)} 帧", flush=True)
    if persist_to_disk:
        pt_path = os.path.join(cache_dir, f"{seq_name}.pt")
        tmp_p = pt_path + ".tmp"
        torch.save(payload, tmp_p)
        os.replace(tmp_p, pt_path)
        del payload
        gc.collect()
        return pt_path
    return payload


def main():
    parser = argparse.ArgumentParser(description="v3 标定组 (calib) 流式低内存训练 HetShield ConvGRU")
    parser.add_argument("--data_dir", type=str, default=OPV2V_TRAIN_DIR)
    parser.add_argument("--split_json", type=str, default=DEFAULT_SPLIT)
    parser.add_argument("--cache_dir", type=str, default=DEFAULT_SEQ_CACHE_DIR)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max_steps_per_seq", type=int, default=65)
    parser.add_argument("--alpha", type=float, default=0.6, help="原论文默认 alpha=0.6")
    parser.add_argument("--energy_thresh", type=float, default=0.5)
    parser.add_argument("--ckpt_dir", type=str, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--out_calib", type=str, default=DEFAULT_OUT_CALIB)
    parser.add_argument("--max_total_frames", type=int, default=0, help=">0 时仅截取前 N 帧用于 2 帧冒烟测试")
    parser.add_argument("--recalib_only", action="store_true")
    args = parser.parse_args()

    t0 = time.time()
    torch.manual_seed(20261120)
    np.random.seed(20261120)

    with open(args.split_json, "r", encoding="utf-8") as f:
        sp = json.load(f)
    if "groups" in sp and "calib" in sp["groups"]:
        fit_seqs = list(sp["groups"]["calib"])
        forbidden_seqs = (
            set(sp["groups"].get("select", []))
            | set(sp["groups"].get("dev", []))
            | set(sp["groups"].get("validate", []))
            | set(sp["groups"].get("test", []))
            | set(sp["groups"].get("test_culvercity", []))
        )
        assert set(fit_seqs).isdisjoint(forbidden_seqs), "硬停机拦截：HetShield calib 序列与其他划分存在重叠！"
    else:
        fit_seqs = list(sp["fit_sequences"].keys())
        eval_seqs = set(sp["eval_sequences"].keys())
        assert set(fit_seqs).isdisjoint(eval_seqs), "Fit 与 Eval 序列存在重叠！"

    is_smoke = int(args.max_total_frames) > 0
    wrapper = HEALBaseWrapper(device="cuda")
    ds_m1 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m1")
    ds_m2 = wrapper.build_dataset(data_dir=args.data_dir, use_cav=2, visualize=True, collab_modality="m2")
    sample_seqs = wrapper.get_sample_sequences(ds_m2)

    seq_to_indices = OrderedDict()
    for i, s in enumerate(sample_seqs):
        if s in fit_seqs:
            seq_to_indices.setdefault(s, []).append(i)

    print(f"[Step 5-1] 按序列准备 calib ({len(fit_seqs)} 序列) 协议特征缓存 (内存峰值 < 3GB)...", flush=True)
    seq_store = OrderedDict()
    budget_rem = int(args.max_total_frames) if is_smoke else None
    active_fit_seqs = []
    for s in fit_seqs:
        if budget_rem is not None and budget_rem <= 0:
            break
        full_idxs = seq_to_indices[s]
        picked = full_idxs[:: max(1, int(args.stride))][: int(args.max_steps_per_seq)]
        if budget_rem is not None:
            picked = picked[: max(2, budget_rem)]
            budget_rem -= len(picked)
        seq_store[s] = _extract_or_load_hs_seq(
            wrapper, ds_m1, ds_m2, s, picked, cache_dir=args.cache_dir, persist_to_disk=not is_smoke
        )
        active_fit_seqs.append(s)
    fit_seqs = active_fit_seqs

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
                entry = _load_seq_entry(seq_store, s)
                idxs_s = entry["indices"]
                n_tr = max(2, int(len(idxs_s) * 0.85))
                n_tr = min(n_tr, len(idxs_s))
                for mod in ("m1", "m2"):
                    cavs = entry[f"{mod}_cav"]
                    vals = entry[f"{mod}_val"]
                    h_s, h_p = None, None
                    for t in range(n_tr - 1):
                        if int(idxs_s[t + 1]) - int(idxs_s[t]) != int(args.stride):
                            h_s, h_p = None, None
                            continue
                        z_prev = cavs[t].cuda().float()
                        z_next = cavs[t + 1].cuda().float()
                        v_next = vals[t + 1].cuda().float()
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
                del entry

            ep_rec = {
                "epoch": ep + 1,
                "shared_mse": round(float(np.mean(ls_all)) if ls_all else 0.0, 6),
                "m1_mse": round(float(np.mean(lp_all["m1"])) if lp_all["m1"] else 0.0, 6),
                "m2_mse": round(float(np.mean(lp_all["m2"])) if lp_all["m2"] else 0.0, 6),
            }
            train_hist.append(ep_rec)
            print(f"  [Epoch {ep + 1}/{args.epochs}] {ep_rec}", flush=True)

        shared_net.eval()
        for m in ("m1", "m2"):
            per_nets[m].eval()
        val_mse = {"shared": [], "m1": [], "m2": [], "identity_baseline": []}
        with torch.no_grad():
            for s in fit_seqs:
                entry = _load_seq_entry(seq_store, s)
                idxs_s = entry["indices"]
                n_tr = int(len(idxs_s) * 0.85)
                if n_tr >= len(idxs_s):
                    n_tr = max(1, len(idxs_s) - 1)
                for mod in ("m1", "m2"):
                    cavs = entry[f"{mod}_cav"]
                    vals = entry[f"{mod}_val"]
                    h_s, h_p = None, None
                    for t in range(len(idxs_s) - 1):
                        z_prev = cavs[t].cuda().float()
                        z_next = cavs[t + 1].cuda().float()
                        v_next = vals[t + 1].cuda().float()
                        denom = torch.clamp(v_next.sum() * float(z_next.shape[1]), min=1.0)
                        pred_s, h_s = shared_net(z_prev, h_s)
                        pred_p, h_p = per_nets[mod](z_prev, h_p)
                        if t >= n_tr - 1:
                            val_mse["shared"].append(float((((pred_s - z_next) ** 2) * v_next).sum().item() / denom.item()))
                            val_mse[mod].append(float((((pred_p - z_next) ** 2) * v_next).sum().item() / denom.item()))
                            val_mse["identity_baseline"].append(float((((z_prev - z_next) ** 2) * v_next).sum().item() / denom.item()))
                del entry

        val_summary = {k: round(float(np.mean(v)), 6) for k, v in val_mse.items() if v}
        os.makedirs(args.ckpt_dir, exist_ok=True)
        torch.save(shared_net.state_dict(), shared_ckpt)
        torch.save({m: per_nets[m].state_dict() for m in ("m1", "m2")}, persensor_ckpt)

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
        def_shared, def_per, seq_store, fit_seqs, stride=int(args.stride), tail_ratio=0.85
    )
    sk_shared = {m: calib_raw["shared"][m]["s_k"] for m in ("m1", "m2")}
    sk_per = {m: calib_raw["persensor"][m]["s_k"] for m in ("m1", "m2")}
    gk_shared = {m: np.asarray(calib_raw["shared"][m]["g_k"], dtype=np.float64) for m in ("m1", "m2")}
    all_sk_shared = sk_shared["m1"] + sk_shared["m2"]

    def _safe_pct(vals_list, q, fallback=0.3):
        return round(float(np.percentile(vals_list, q)), 6) if len(vals_list) > 0 else float(fallback)

    calib = OrderedDict({
        "meta": {
            "claim": "[临时结论 · 待人工代码复核]",
            "description": "v3 标定组 (calib) 流式训练 HetShield ConvGRU（alpha=0.6，协议特征层单次向本车混合，论文默认 0.3 版不调参）",
            "calib_sequences": fit_seqs,
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
            "tuned": False,
            "mitigation": "protocol_layer_once: z_tilde_k = g_k * z_k + (1 - g_k) * z_0",
        },
        "fit_tuned_shared_tau_p95": _safe_pct(all_sk_shared, 95.0, 0.45),
        "fit_tuned_shared_tau_p98": _safe_pct(all_sk_shared, 98.0, 0.60),
        "fit_persensor_tau_p95": {
            "m1": _safe_pct(sk_per["m1"], 95.0, 0.45),
            "m2": _safe_pct(sk_per["m2"], 95.0, 0.55),
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
            "shared_m1": {k: _safe_pct(sk_shared["m1"], q) for k, q in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))},
            "shared_m2": {k: _safe_pct(sk_shared["m2"], q) for k, q in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))},
            "persensor_m1": {k: _safe_pct(sk_per["m1"], q) for k, q in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))},
            "persensor_m2": {k: _safe_pct(sk_per["m2"], q) for k, q in (("p50", 50), ("p90", 90), ("p95", 95), ("p99", 99))},
        },
    })
    os.makedirs(os.path.dirname(args.out_calib), exist_ok=True)
    with open(args.out_calib, "w", encoding="utf-8") as f:
        json.dump(calib, f, indent=2, ensure_ascii=False)
    print(f"\n[完成] HetShield 权重与初始统计已保存至：{args.out_calib} (总耗时 {time.time() - t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
