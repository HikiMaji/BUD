#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# 全自动容错实验守护流水线（严格零泄露：仅运行 FIT-11 / DEV-8，绝不触碰 opv2v_test）
# 规范：
#   1. 单步容错：某步失败绝不阻断不依赖它的独立步骤；仅在强依赖前置文件缺失时跳过该依赖步；
#   2. 每步开始前强制校验 git 工作区无未提交的代码改动；
#   3. 每步产出结果 JSON 后立即写入当前 git commit hash；
#   4. 每步完成后自动提交代码和结果 JSON（不提交日志、权重、缓存、图像）。

set -uo pipefail

REPO="/root/autodl-tmp/proj_dir2_hetero_trust"
PY="/root/miniconda3/bin/python"
cd "$REPO"
mkdir -p logs

export PYTHONUNBUFFERED=1

SUMMARY_FILE="logs/pipeline_status_summary.txt"
: > "$SUMMARY_FILE"

record_status() {
    local step_name="$1"
    local status="$2"
    local detail="$3"
    echo "[$(date '+%F %T')] ${step_name} : ${status} (${detail})" | tee -a "$SUMMARY_FILE"
}

# 用法: run_step <step_name> <log_file> <out_json> <comma_sep_req_files_or_NONE> <cmd...>
run_step() {
    local step_name="$1"
    local log_file="$2"
    local out_json="$3"
    local req_files="$4"
    shift 4

    echo "============================================================"
    echo "[$(date '+%F %T')] >>> 准备步骤: $step_name"
    echo "============================================================"

    # 0. 检查强依赖前置文件是否存在
    if [ "$req_files" != "NONE" ]; then
        IFS=',' read -ra REQS <<< "$req_files"
        for rf in "${REQS[@]}"; do
            if [ ! -f "$rf" ]; then
                echo "[SKIP] 缺少前置依赖文件: $rf，跳过 $step_name"
                record_status "$step_name" "SKIP_DEP" "missing $rf"
                return 0
            fi
        done
    fi

    # 1. 开始前校验工作区干净
    if ! "$PY" src_v2_mm/git_provenance.py --repo "$REPO" --check-clean; then
        echo "[FAIL] git 工作区存在未提交改动，终止步骤 $step_name"
        record_status "$step_name" "FAIL_DIRTY_GIT" "workspace dirty"
        return 0
    fi

    # 2. 执行实验脚本（单步捕获返回码，不中断后续独立任务）
    local t_start
    t_start=$(date +%s)
    "$@" 2>&1 | tee "$log_file"
    local rc=${PIPESTATUS[0]}
    local t_end
    t_end=$(date +%s)
    local dur=$((t_end - t_start))

    if [ "$rc" -ne 0 ]; then
        echo "[FAIL] 步骤 $step_name 异常退出 (rc=$rc, ${dur}s)，继续执行后续无依赖步骤！"
        record_status "$step_name" "FAIL" "rc=${rc}, ${dur}s"
        return 0
    fi

    # 3. 注入 git commit 并仅提交代码与结果 JSON
    if [ -f "$out_json" ]; then
        "$PY" src_v2_mm/git_provenance.py --repo "$REPO" --inject-json "$out_json"
        "$PY" src_v2_mm/git_provenance.py --repo "$REPO" \
            --auto-commit "[auto] 完成 ${step_name}" \
            --result-files "$out_json"
        record_status "$step_name" "PASS" "${dur}s -> ${out_json}"
    else
        echo "[FAIL] 未找到预期结果文件: $out_json (${dur}s)"
        record_status "$step_name" "FAIL_NO_OUT" "missing ${out_json}, ${dur}s"
    fi
    return 0
}

# Step 1: 步骤 ④-4 纯 Fit 盲区修复与标定（新物理可见性定义 + 前景/空地双分档 + 排除本车自身框）
if [ "${SKIP_STEP4_4:-0}" = "1" ] && [ -f "results_rq1_v2_mm/step4_4_bud_frozen_calibrator.json" ]; then
    record_status "step4_4_fit_blind_recalib" "SKIP_EXISTING" "SKIP_STEP4_4=1"
else
    run_step "step4_4_fit_blind_recalib" \
        "logs/overnight_01_step4_4.log" \
        "results_rq1_v2_mm/step4_4_bud_frozen_calibrator.json" \
        "NONE" \
        "$PY" scripts_v2_mm/step4_4_fit_blind_diag_and_recalib.py
fi

# Step 2: 步骤 ⑤-1 HetShield 64ch ConvGRU 训练与 FIT-11 标定（不依赖 step4_4）
run_step "step5_1_train_hetshield" \
    "logs/overnight_02_step5_1.log" \
    "results_rq1_v2_mm/step5_1_hetshield_heal_calib.json" \
    "NONE" \
    "$PY" scripts_v2_mm/step5_1_train_hetshield_heal.py

# Step 3: 【最高优先级立足点实验】步骤 ⑤-3 异构车队统一判据跨模态误报与检出权衡实证
run_step "step5_3_hetero_unified_evidence" \
    "logs/overnight_03_step5_3_evidence.log" \
    "results_rq1_v2_mm/step5_3_hetero_unified_threshold_evidence.json" \
    "results_rq1_v2_mm/step4_4_bud_frozen_calibrator.json,results_rq1_v2_mm/step5_1_hetshield_heal_calib.json" \
    "$PY" scripts_v2_mm/step5_3_hetero_unified_threshold_evidence.py

# Step 4: 低成本补充实验（研究点二前置：m1/m2 径向 vs 横向框中心误差各向异性分解，零依赖）
run_step "step6_aniso_box_error" \
    "logs/overnight_04_step6_aniso.log" \
    "results_rq1_v2_mm/step6_aniso_box_error_stats.json" \
    "NONE" \
    "$PY" scripts_v2_mm/step6_aniso_box_error_analysis.py

# Step 5: 步骤 ⑥ 限时探路 m3 (32 线 LiDAR + SECOND，零依赖)
run_step "step6_probe_m3" \
    "logs/overnight_05_step6_m3.log" \
    "results_rq1_v2_mm/step6_m3_probe_report.json" \
    "NONE" \
    "$PY" scripts_v2_mm/step6_probe_m3_second.py

# Step 6: 步骤 ④-5 BUD-Fusion Dev-320 全维度验收（9 工况 + 成对 Bootstrap 95% CI + 遮挡伪造修复前后对比）
run_step "step4_5_eval_bud_frozen_320" \
    "logs/overnight_06_step4_5.log" \
    "results_rq1_v2_mm/step4_5_bud_frozen_320_report.json" \
    "results_rq1_v2_mm/step4_4_bud_frozen_calibrator.json" \
    "$PY" scripts_v2_mm/step4_5_eval_bud_frozen_320.py

# Step 7: 步骤 ⑦ 单帧耗时拆解（CUDA Event 精确计时）
run_step "step7_latency_breakdown" \
    "logs/overnight_07_step7_latency.log" \
    "results_rq1_v2_mm/step7_latency_breakdown.json" \
    "results_rq1_v2_mm/step4_4_bud_frozen_calibrator.json" \
    "$PY" scripts_v2_mm/step7_latency_breakdown.py

# Step 8: 任务 3 多车队分层主实验 + 全基线同台（mixed 主实验优先 -> all_m1 -> all_m2）
run_step "step5_2_multicav_and_baselines" \
    "logs/overnight_08_step5_2_multicav.log" \
    "results_rq1_v2_mm/step5_2_multicav_baselines_report.json" \
    "results_rq1_v2_mm/step4_4_bud_frozen_calibrator.json,results_rq1_v2_mm/step5_1_hetshield_heal_calib.json" \
    "$PY" scripts_v2_mm/step5_2_eval_multicav_and_baselines.py --fleets mixed all_m1 all_m2

echo "============================================================"
echo "[$(date '+%F %T')] === 全自动容错流水线执行完毕（未触碰 opv2v_test）==="
cat "$SUMMARY_FILE"
echo "============================================================"
