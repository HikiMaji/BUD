#!/usr/bin/env bash
set -eo pipefail
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
export PYTHONPATH=/root/autodl-tmp/proj_dir2_hetero_trust:/root/autodl-tmp/proj_dir2_hetero_trust/third_party/OpenCOOD
cd /root/autodl-tmp/proj_dir2_hetero_trust
echo "[$(date)] Starting Step 3 (150 Unseen Test Frames)..."
/root/miniconda3/bin/python -u scripts/step3_evaluate_all.py --max_test_frames 150
echo "[$(date)] Starting Step 4 (Quantitative Figures)..."
/root/miniconda3/bin/python -u scripts/step4_plot_figures.py
echo "[$(date)] Starting Step 5 (Qualitative BEV Case Figure)..."
/root/miniconda3/bin/python -u scripts/step5_visualize_bev_case.py
echo "[$(date)] ALL STEPS SUCCEEDED!"
