# -*- coding: utf-8 -*-
"""
M1 / A6：构建全量原版 PNG OPV2V 数据集序列清单与全新分层随机划分（results_v3/splits_v3.json）。

严格遵守《overnight_m1_m8_gate_plan.md》第一节第 1 条与 M1 全部规范：
1. 彻底摒弃旧 Fit-11 / Dev-8 分组与旧 train_full19_444 / Dev-320；
2. 扫描 /root/autodl-tmp/datasets/OPV2V/train 下全部序列，记录序列名、帧数、总车数、城镇地图簇与相机数，
   自动排除只有 3 个相机的序列（如 2021_09_09_13_20_58）；
3. 全部有效 4 相机训练序列（43 条）按“总车数（2 / 3 / 4 / >=5）× 城镇”分层随机（固定种子 20261120），
   划分为 calib（约 60% 帧）、select（约 25% 帧）、dev（约 15% 帧），且保证每组均包含总车数 >=4 的序列；
4. 强制断言 calib / select / dev 三组之间、以及它们与 validate / test / test_culvercity 之间完全无重叠；
5. 生成固定种子共用的抽帧索引：
   - calib   : 全量帧（>= 3,000 帧）；
   - select  : 每条序列等间隔抽样，总计 600 帧（覆盖全部 select 序列）；
   - dev_fast: 每条序列等间隔抽样，总计 400 帧（覆盖全部 dev 序列）；
   - dev_full: dev 组全部帧（用于冻结前最终确认）。
"""

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, OrderedDict
from typing import Dict, List, Tuple

import numpy as np
import yaml

PROJ_ROOT = "/root/autodl-tmp/proj_dir2_hetero_trust"
if os.path.exists(PROJ_ROOT) and PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)
LOCAL_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if LOCAL_ROOT not in sys.path:
    sys.path.insert(0, LOCAL_ROOT)

from src_v2_mm.step4_1_v2_core import (
    OPV2V_TEST_CULVERCITY_DIR,
    OPV2V_TEST_DIR,
    OPV2V_TRAIN_DIR,
    OPV2V_VALIDATE_DIR,
)

SPLIT_SEED = 20261120


def infer_carla_town_from_metadata(seq_dir: str, cav_ids: List[str]) -> Tuple[str, dict]:
    """
    根据序列首帧车辆世界坐标 lidar_pose 与 data_protocol.yaml 中的道路范围/限速客观判定所属 CARLA 城镇地图区域。
    CARLA 各城镇的世界坐标系区间互不相同（如 Town06 高速 X 延伸至 +630m，Town04 带 Z>7m 立交与 X=-450m/-390m 环路，
    Town05 西侧外环 X∈[-285,-180]，Town03 隧道与城区 (203,-250) 等）。
    """
    first_cav_dir = os.path.join(seq_dir, cav_ids[0])
    yaml_files = sorted(
        f for f in os.listdir(first_cav_dir)
        if f.endswith(".yaml") and "additional" not in f and not f.startswith("data_protocol")
    )
    pose_xyz = [0.0, 0.0, 0.0]
    if yaml_files:
        with open(os.path.join(first_cav_dir, yaml_files[0]), "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        m = re.search(r"lidar_pose:\s*\n((?:\s*-\s*[-0-9.eE+]+\s*\n){3})", txt)
        if m:
            raw_lines = [ln.strip()[1:].strip() for ln in m.group(1).strip().splitlines()]
            pose_xyz = [float(v) for v in raw_lines[:3]]

    max_speed = None
    tm_range = None
    proto_path = os.path.join(seq_dir, "data_protocol.yaml")
    if os.path.exists(proto_path):
        try:
            with open(proto_path, "r", encoding="utf-8", errors="ignore") as f:
                pd = yaml.safe_load(f) or {}
            if isinstance(pd.get("world"), dict) and pd["world"].get("town"):
                return str(pd["world"]["town"]), {"pose0_xyz": pose_xyz, "source": "world.town"}
            tm_range = (pd.get("carla_traffic_manager") or {}).get("range")
            scl = (pd.get("scenario") or {}).get("single_cav_list") or []
            if scl and isinstance(scl[0], dict):
                max_speed = (scl[0].get("behavior") or {}).get("max_speed")
        except Exception:
            pass

    x0, y0, z0 = pose_xyz
    r0 = tm_range[0][:4] if (tm_range and len(tm_range) > 0 and len(tm_range[0]) >= 4) else None

    if x0 >= 450.0 or (r0 is not None and float(r0[1]) >= 450.0):
        town = "Town06"
    elif z0 >= 6.0 or y0 <= -300.0 or (r0 is not None and (float(r0[0]) <= -300.0 or float(r0[2]) <= -300.0)) or (max_speed is not None and float(max_speed) >= 65):
        town = "Town04"
    elif x0 <= -175.0 or (r0 is not None and float(r0[0]) <= -185.0):
        town = "Town05"
    elif y0 >= 260.0 or (r0 is not None and float(r0[3]) >= 260.0):
        town = "Town01"
    elif (0.0 <= x0 <= 70.0 and 120.0 <= y0 <= 220.0) and (max_speed is None or float(max_speed) <= 35):
        town = "Town02"
    elif (x0 >= 180.0 and y0 <= -200.0) or (r0 is not None and float(r0[3]) >= 350.0):
        town = "Town03"
    elif 50.0 <= x0 <= 115.0 and -30.0 <= y0 <= 95.0 and (max_speed is not None and float(max_speed) <= 35):
        town = "Town10HD"
    elif max_speed is not None and float(max_speed) >= 50:
        town = "Town07"
    else:
        town = "Town03"

    meta = {
        "pose0_xyz": [round(v, 2) for v in pose_xyz],
        "max_speed": max_speed,
        "tm_range": r0,
    }
    return town, meta


def scan_split_inventory(split_dir: str) -> Tuple[List[dict], List[dict]]:
    """扫描单个划分目录下的全部序列，返回 (valid_4cam_inventory, excluded_sequences)。"""
    if not os.path.isdir(split_dir):
        return [], []
    valid_list = []
    excluded_list = []

    for seq_name in sorted(os.listdir(split_dir)):
        seq_dir = os.path.join(split_dir, seq_name)
        if not os.path.isdir(seq_dir):
            continue
        cav_ids = sorted(c for c in os.listdir(seq_dir) if os.path.isdir(os.path.join(seq_dir, c)))
        if not cav_ids:
            excluded_list.append({"seq_name": seq_name, "reason": "no_cav_directories"})
            continue

        # 统计 ego 首车帧数及所有车辆的相机数完整性
        ego_cid = cav_ids[0] if cav_ids[0] != "-1" or len(cav_ids) == 1 else cav_ids[1]
        ego_pcds = sorted(f[:-4] for f in os.listdir(os.path.join(seq_dir, ego_cid)) if f.endswith(".pcd"))
        num_frames = len(ego_pcds)

        min_cams = 999
        max_cams = 0
        tot_pcd = 0
        tot_png = 0
        all_4cam_ok = num_frames > 0

        for cid in cav_ids:
            cdir = os.path.join(seq_dir, cid)
            fs = set(os.listdir(cdir))
            pcds = [f[:-4] for f in fs if f.endswith(".pcd")]
            pngs = [f for f in fs if f.endswith(".png") and "_camera" in f]
            tot_pcd += len(pcds)
            tot_png += len(pngs)
            if not pcds:
                all_4cam_ok = False
                min_cams = 0
                continue
            for ts in pcds:
                c_cnt = sum(1 for k in range(6) if f"{ts}_camera{k}.png" in fs)
                min_cams = min(min_cams, c_cnt)
                max_cams = max(max_cams, c_cnt)
                if not all(f"{ts}_camera{k}.png" in fs for k in range(4)):
                    all_4cam_ok = False

        town, town_meta = infer_carla_town_from_metadata(seq_dir, cav_ids)
        rec = OrderedDict({
            "seq_name": seq_name,
            "num_frames": int(num_frames),
            "num_cavs": len(cav_ids),
            "town": town,
            "num_cameras": int(min_cams if min_cams != 999 else 0),
            "total_pcd_files": int(tot_pcd),
            "total_camera_png_files": int(tot_png),
            "town_meta": town_meta,
        })
        if all_4cam_ok and min_cams == 4 and max_cams == 4:
            valid_list.append(rec)
        else:
            rec["exclude_reason"] = f"cameras_range=[{min_cams},{max_cams}], all_4cam_ok={all_4cam_ok}"
            excluded_list.append(rec)

    return valid_list, excluded_list


def stratified_split_all_train_sequences(
    inventory: List[dict],
    seed: int = SPLIT_SEED,
    target_ratios: Tuple[float, float, float] = (0.60, 0.25, 0.15),
) -> Dict[str, List[dict]]:
    """
    全新分层随机划分全部 4 相机训练序列（完全不沿用旧 Fit-11 / Dev-8 捆绑）：
    - 分层依据：总车数档位（"2", "3", "4", ">=5"）× 城镇（town）；
    - 随机种子固定 20261120；
    - 强制保证 calib / select / dev 三组每一组都包含总车数 >= 4 的序列，且总帧数比例逼近 60% / 25% / 15%。
    """
    rng = random.Random(int(seed))
    total_frames = sum(item["num_frames"] for item in inventory)
    target_frames = {
        "calib": target_ratios[0] * total_frames,
        "select": target_ratios[1] * total_frames,
        "dev": target_ratios[2] * total_frames,
    }
    groups: Dict[str, List[dict]] = {"calib": [], "select": [], "dev": []}

    def _cav_tier(n_cav: int) -> str:
        return ">=5" if n_cav >= 5 else str(n_cav)

    # 1) 先把 >=4 车的序列按 (cav_tier, town) 分层打乱，确保每组至少先分得 1 条 >=4 车序列，其余按比例缺口分配
    ge4_items = [it for it in inventory if it["num_cavs"] >= 4]
    lt4_items = [it for it in inventory if it["num_cavs"] < 4]
    assert len(ge4_items) >= 3, f"训练集中总车数 >=4 的序列不足 3 条（实际 {len(ge4_items)}），无法满足每组均含 >=4 车！"

    ge4_strata: Dict[Tuple[str, str], List[dict]] = {}
    for it in ge4_items:
        ge4_strata.setdefault((_cav_tier(it["num_cavs"]), it["town"]), []).append(it)
    ge4_shuffled = []
    for k in sorted(ge4_strata.keys()):
        b = list(ge4_strata[k])
        rng.shuffle(b)
        ge4_shuffled.extend(b)
    rng.shuffle(ge4_shuffled)

    # 保底每组各 1 条 >=4 车序列（为使 select / dev 多车验证充足，若有 >=6 条则给 select 与 dev 各保底 2 条）
    min_ge4_per_group = {"calib": 2, "select": 2, "dev": 2} if len(ge4_shuffled) >= 6 else {"calib": 1, "select": 1, "dev": 1}
    idx_ptr = 0
    for g_name in ("dev", "select", "calib"):
        for _ in range(min_ge4_per_group[g_name]):
            groups[g_name].append(ge4_shuffled[idx_ptr])
            idx_ptr += 1
    for it in ge4_shuffled[idx_ptr:]:
        cur_f = {g: sum(x["num_frames"] for x in groups[g]) for g in groups}
        best_g = max(("calib", "select", "dev"), key=lambda g: (target_frames[g] - cur_f[g]) / target_frames[g])
        groups[best_g].append(it)

    # 2) 对 2 车与 3 车序列按 (cav_tier, town) 分层打乱，按各组帧数缺口比例均衡分配
    lt4_strata: Dict[Tuple[str, str], List[dict]] = {}
    for it in lt4_items:
        lt4_strata.setdefault((_cav_tier(it["num_cavs"]), it["town"]), []).append(it)

    for k in sorted(lt4_strata.keys()):
        bucket = list(lt4_strata[k])
        rng.shuffle(bucket)
        for it in bucket:
            cur_f = {g: sum(x["num_frames"] for x in groups[g]) for g in groups}
            best_g = max(("calib", "select", "dev"), key=lambda g: (target_frames[g] - cur_f[g]) / target_frames[g])
            groups[best_g].append(it)

    for g in groups:
        groups[g].sort(key=lambda x: x["seq_name"])
    return groups


def allocate_proportional_counts(seq_lengths: Dict[str, int], target_total: int, min_per_seq: int = 8) -> Dict[str, int]:
    """按序列长度比例精确分配抽帧数，保证总和严格等于 target_total 且覆盖每一条序列（每序列 >= min_per_seq 或其全长）。"""
    seq_names = list(seq_lengths.keys())
    tot_avail = sum(seq_lengths.values())
    if target_total >= tot_avail:
        return dict(seq_lengths)

    counts = {}
    remainders = []
    for s in seq_names:
        cap = seq_lengths[s]
        floor_min = min(cap, int(min_per_seq))
        exact = float(target_total) * float(cap) / float(tot_avail)
        base = max(floor_min, min(cap, int(exact)))
        counts[s] = base
        remainders.append((exact - base, s))

    diff = int(target_total) - sum(counts.values())
    if diff > 0:
        remainders.sort(key=lambda x: (-x[0], x[1]))
        while diff > 0:
            progressed = False
            for _, s in remainders:
                if diff == 0:
                    break
                if counts[s] < seq_lengths[s]:
                    counts[s] += 1
                    diff -= 1
                    progressed = True
            if not progressed:
                break
    elif diff < 0:
        remainders.sort(key=lambda x: (x[0], x[1]))
        while diff < 0:
            progressed = False
            for _, s in remainders:
                if diff == 0:
                    break
                floor_min = min(seq_lengths[s], int(min_per_seq))
                if counts[s] > floor_min:
                    counts[s] -= 1
                    diff += 1
                    progressed = True
            if not progressed:
                break
    return counts


def build_even_subsample_subset(
    group_seqs: List[str],
    seq_to_global_indices: Dict[str, List[int]],
    target_total: int,
    min_per_seq: int = 10,
) -> OrderedDict:
    """生成兼容全仓评测器格式的均衡抽帧子集字典（seq_order, total_frames, all_indices, sequences）。"""
    lengths = {s: len(seq_to_global_indices[s]) for s in group_seqs}
    alloc = allocate_proportional_counts(lengths, target_total=target_total, min_per_seq=min_per_seq)

    sequences_dict = OrderedDict()
    all_picked = []
    for s in group_seqs:
        full_idxs = seq_to_global_indices[s]
        k = alloc[s]
        if k >= len(full_idxs):
            picked = [int(i) for i in full_idxs]
        else:
            pos = np.linspace(0, len(full_idxs) - 1, k, dtype=int)
            # 去重补足防止极少数四舍五入碰撞
            seen = set()
            picked = []
            for p in pos:
                gi = int(full_idxs[int(p)])
                if gi not in seen:
                    seen.add(gi)
                    picked.append(gi)
            if len(picked) < k:
                for gi in full_idxs:
                    if int(gi) not in seen:
                        seen.add(int(gi))
                        picked.append(int(gi))
                        if len(picked) == k:
                            break
            picked.sort()
        all_picked.extend(picked)
        sequences_dict[s] = {
            "total_frames": len(full_idxs),
            "sampled_frames": len(picked),
            "indices": picked,
        }

    return OrderedDict({
        "seq_order": list(group_seqs),
        "num_sequences": len(group_seqs),
        "total_frames": len(all_picked),
        "all_indices": all_picked,
        "sequences": sequences_dict,
    })


def main():
    parser = argparse.ArgumentParser(description="M1: 生成全量 43 序列分层划分 results_v3/splits_v3.json")
    parser.add_argument("--train_dir", type=str, default=OPV2V_TRAIN_DIR)
    parser.add_argument("--val_dir", type=str, default=OPV2V_VALIDATE_DIR)
    parser.add_argument("--test_dir", type=str, default=OPV2V_TEST_DIR)
    parser.add_argument("--culver_dir", type=str, default=OPV2V_TEST_CULVERCITY_DIR)
    parser.add_argument("--select_frames", type=int, default=600)
    parser.add_argument("--dev_fast_frames", type=int, default=400)
    parser.add_argument("--verify_with_opencood", action="store_true", help="使用 OpenCOOD Dataset 核验全局帧索引")
    parser.add_argument("--out_json", type=str, default=os.path.join(PROJ_ROOT, "results_v3", "splits_v3.json"))
    args = parser.parse_args()

    train_inv, train_excl = scan_split_inventory(args.train_dir)
    val_inv, val_excl = scan_split_inventory(args.val_dir)
    test_inv, test_excl = scan_split_inventory(args.test_dir)
    culver_inv, culver_excl = scan_split_inventory(args.culver_dir)

    assert len(train_excl) == 0, f"OPV2V/train 目录中存在未排除的异常或 3 相机序列：{train_excl}"
    assert len(train_inv) >= 40, f"有效 4 相机训练序列数异常：{len(train_inv)}"

    # 若启用 OpenCOOD 核验，直接与真实 Dataset 的 scenario_folders / len_record 对齐全局下标
    seq_to_global_indices: Dict[str, List[int]] = OrderedDict()
    if args.verify_with_opencood:
        from src_v2_mm.heal_base_wrapper import HEALBaseWrapper
        wrapper = HEALBaseWrapper(data_dir=args.train_dir, use_cav=2, device="cpu")
        ds = wrapper.build_dataset(data_dir=args.train_dir, use_cav=2, visualize=False, collab_modality="m2")
        sample_seqs = wrapper.get_sample_sequences(ds)
        for gi, s_name in enumerate(sample_seqs):
            seq_to_global_indices.setdefault(s_name, []).append(int(gi))
        for rec in train_inv:
            s = rec["seq_name"]
            assert s in seq_to_global_indices, f"OpenCOOD 未加载训练序列 {s}"
            rec["num_frames"] = len(seq_to_global_indices[s])
    else:
        cursor = 0
        for rec in train_inv:
            n_f = int(rec["num_frames"])
            seq_to_global_indices[rec["seq_name"]] = list(range(cursor, cursor + n_f))
            cursor += n_f

    groups = stratified_split_all_train_sequences(train_inv, seed=SPLIT_SEED)

    calib_seqs = [x["seq_name"] for x in groups["calib"]]
    select_seqs = [x["seq_name"] for x in groups["select"]]
    dev_seqs = [x["seq_name"] for x in groups["dev"]]

    s_cal, s_sel, s_dev = set(calib_seqs), set(select_seqs), set(dev_seqs)
    s_val = {x["seq_name"] for x in val_inv} | {x["seq_name"] for x in val_excl}
    s_tst = {x["seq_name"] for x in test_inv} | {x["seq_name"] for x in test_excl}
    s_cul = {x["seq_name"] for x in culver_inv} | {x["seq_name"] for x in culver_excl}

    # 硬停机红线 ③：严格断言所有集合两两零重叠！
    assert s_cal.isdisjoint(s_sel), f"calib 与 select 重叠: {s_cal & s_sel}"
    assert s_cal.isdisjoint(s_dev), f"calib 与 dev 重叠: {s_cal & s_dev}"
    assert s_sel.isdisjoint(s_dev), f"select 与 dev 重叠: {s_sel & s_dev}"
    s_train_all = s_cal | s_sel | s_dev
    assert len(s_train_all) == len(train_inv), "三组并集未完整覆盖全部有效训练序列！"
    assert s_train_all.isdisjoint(s_val), f"train 与 validate 重叠: {s_train_all & s_val}"
    assert s_train_all.isdisjoint(s_tst), f"train 与 test 重叠: {s_train_all & s_tst}"
    assert s_val.isdisjoint(s_tst), f"validate 与 test 重叠: {s_val & s_tst}"
    if s_cul:
        assert s_train_all.isdisjoint(s_cul), f"train 与 test_culvercity 重叠: {s_train_all & s_cul}"
        assert s_val.isdisjoint(s_cul), f"validate 与 test_culvercity 重叠: {s_val & s_cul}"
        assert s_tst.isdisjoint(s_cul), f"test 与 test_culvercity 重叠: {s_tst & s_cul}"

    # 强制断言每组都包含总车数 >=4 的序列
    for g_name in ("calib", "select", "dev"):
        n_ge4 = sum(1 for x in groups[g_name] if x["num_cavs"] >= 4)
        assert n_ge4 >= 1, f"{g_name} 组缺少总车数 >=4 的序列！"

    calib_total_f = sum(x["num_frames"] for x in groups["calib"])
    assert calib_total_f >= 3000, f"calib 组帧数 {calib_total_f} < 3000！"

    calib_subset = build_even_subsample_subset(calib_seqs, seq_to_global_indices, target_total=calib_total_f, min_per_seq=1)
    select_subset = build_even_subsample_subset(select_seqs, seq_to_global_indices, target_total=int(args.select_frames), min_per_seq=12)
    dev_fast_subset = build_even_subsample_subset(dev_seqs, seq_to_global_indices, target_total=int(args.dev_fast_frames), min_per_seq=12)
    dev_full_total = sum(x["num_frames"] for x in groups["dev"])
    dev_full_subset = build_even_subsample_subset(dev_seqs, seq_to_global_indices, target_total=dev_full_total, min_per_seq=1)

    tot_train_f = sum(x["num_frames"] for x in train_inv)
    summary_stats = OrderedDict()
    for g_name, g_items, sub in (
        ("calib", groups["calib"], calib_subset),
        ("select", groups["select"], select_subset),
        ("dev", groups["dev"], dev_full_subset),
    ):
        gf = sum(x["num_frames"] for x in g_items)
        cav_hist = Counter(x["num_cavs"] for x in g_items)
        town_hist = Counter(x["town"] for x in g_items)
        summary_stats[g_name] = OrderedDict({
            "num_sequences": len(g_items),
            "total_frames": gf,
            "frame_ratio": round(gf / float(tot_train_f), 4),
            "sampled_frames": sub["total_frames"] if g_name != "dev" else {"fast": dev_fast_subset["total_frames"], "full": dev_full_subset["total_frames"]},
            "num_ge4_cav_sequences": sum(1 for x in g_items if x["num_cavs"] >= 4),
            "cav_count_distribution": {str(k): int(v) for k, v in sorted(cav_hist.items())},
            "town_distribution": dict(sorted(town_hist.items())),
        })

    out_obj = OrderedDict({
        "meta": {
            "description": "OPV2V 全量原版 PNG 数据集序列清单与全新分层随机划分（calib / select / dev）",
            "seed": SPLIT_SEED,
            "stratification": "num_cavs_tier (2 / 3 / 4 / >=5) x carla_town",
            "legacy_fit11_dev8_used": False,
            "image_format": "lossless_png",
            "paths": {
                "train": args.train_dir,
                "validate": args.val_dir,
                "test": args.test_dir,
                "test_culvercity": args.culver_dir,
                "test_culvercity_present": bool(len(culver_inv) > 0),
            },
            "disjoint_assertions_passed": True,
            "verified_with_opencood_dataset": bool(args.verify_with_opencood),
            "total_valid_train_sequences": len(train_inv),
            "total_valid_train_frames": tot_train_f,
            "validate_sequences": len(val_inv),
            "validate_frames": sum(x["num_frames"] for x in val_inv),
            "test_sequences": len(test_inv),
            "test_frames": sum(x["num_frames"] for x in test_inv),
        },
        "group_summary": summary_stats,
        "groups": {
            "calib": calib_seqs,
            "select": select_seqs,
            "dev": dev_seqs,
            "validate": [x["seq_name"] for x in val_inv],
            "test": [x["seq_name"] for x in test_inv],
            "test_culvercity": [x["seq_name"] for x in culver_inv],
        },
        "subsets": {
            "calib": calib_subset,
            "select": select_subset,
            "dev_fast": dev_fast_subset,
            "dev_full": dev_full_subset,
        },
        "inventory": {
            "train": train_inv,
            "train_excluded": train_excl,
            "validate": val_inv,
            "test": test_inv,
            "test_culvercity": culver_inv,
        },
    })

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    tmp_p = args.out_json + ".tmp"
    with open(tmp_p, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp_p, args.out_json)

    print(f"[M1 完成] splits_v3.json 已生成: {args.out_json}")
    for g, st in summary_stats.items():
        print(f"  - {g:6s}: seqs={st['num_sequences']:2d}, frames={st['total_frames']:4d} ({st['frame_ratio']*100:.1f}%), ge4_seqs={st['num_ge4_cav_sequences']}, sampled={st['sampled_frames']}")


if __name__ == "__main__":
    main()
