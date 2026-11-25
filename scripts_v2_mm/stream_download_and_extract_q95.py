# -*- coding: utf-8 -*-
"""
服务器端安全流式下载与控盘提取脚本 (stream_download_and_extract_q95.py)
专为 AutoDL 56GB 磁盘限制设计：
- 激活学术加速 (source /etc/network_turbo)
- 单分卷下载 -> 内存/临时解压 -> 仅抽取 42 条序列相机图片转为 Q95 JPG -> 立即彻底删除原压缩包
- 全程监控磁盘剩余容量，低于 30GB 自动硬熔断，绝不爆盘！

使用方式（在服务器另开终端执行）：
1. 优先提取验证集（最快，仅需 ~1.61 GB 空间，5~8 分钟完成）：
   source /etc/network_turbo
   python scripts_v2_mm/stream_download_and_extract_q95.py --val_only

2. 全量提取 42 条序列（总计 9.52 GB）：
   source /etc/network_turbo
   python scripts_v2_mm/stream_download_and_extract_q95.py --all
"""

import os
import sys
import glob
import shutil
import zipfile
import argparse
import subprocess
from PIL import Image

TARGET_SEQUENCES = {
    "validate": [
        "2021_08_22_13_37_16", "2021_08_22_22_01_17", "2021_08_23_10_51_24",
        "2021_08_23_13_17_21", "2021_08_23_19_42_07", "2021_09_09_19_27_35"
    ],
    "test": [
        "2021_08_18_19_48_05", "2021_08_20_21_10_24", "2021_08_21_09_28_12", "2021_08_22_07_52_02",
        "2021_08_22_09_08_29", "2021_08_22_21_41_24", "2021_08_23_12_58_19", "2021_08_23_15_19_19",
        "2021_08_23_16_06_26", "2021_08_23_17_22_47", "2021_08_23_21_07_10", "2021_08_23_21_47_19",
        "2021_08_24_07_45_41", "2021_08_24_11_37_54", "2021_08_24_20_09_18", "2021_08_24_20_49_54"
    ],
    "train": [
        "2021_08_16_22_26_54", "2021_08_18_09_02_56", "2021_08_18_19_11_02", "2021_08_18_22_16_12",
        "2021_08_20_20_39_00", "2021_08_21_09_09_41", "2021_08_21_15_41_04", "2021_08_21_22_21_37",
        "2021_08_22_06_43_37", "2021_08_22_07_24_12", "2021_08_22_10_10_40", "2021_08_22_10_46_58",
        "2021_08_22_22_30_58", "2021_08_23_11_22_46", "2021_08_23_23_08_17", "2021_08_24_21_29_28",
        "2021_09_09_13_20_58", "2021_09_09_22_21_11", "2021_09_09_23_21_21", "2021_09_10_12_07_11"
    ]
}

# 官方 Google Drive 分卷文件 ID 映射（来自 OpenCOOD 官方 download.sh）
GDRIVE_FILE_IDS = {
    "validate.zip": "1KW1Ya8CGvJuqB5dUBmH968zDFbBFgNtm",  # data_experiment.zip / validate
    "test.zip": "11JdX113eVs6wtCS0c5j7L6KfQcJQn2h0",
}


def check_disk_space(target_path="/root/autodl-tmp", min_free_gb=30.0):
    st = os.statvfs(target_path)
    free_gb = (st.f_bavail * st.f_frsize) / (1024 ** 3)
    if free_gb < min_free_gb:
        raise RuntimeError(f"[ABORT] 磁盘可用空间仅剩 {free_gb:.1f} GB (< {min_free_gb} GB 安全阈值)！紧急终止以防爆盘！")
    return free_gb


def process_zip_to_q95(zip_path, output_base_dir, quality=95):
    """直接流式读取 zip 内容，仅将目标序列的 camera0~3.png 转换并落盘为 JPG，零冗余！"""
    print(f"[*] 正在从压缩包 {os.path.basename(zip_path)} 中流式筛选并转换目标图像...", flush=True)
    all_target_seqs = set(TARGET_SEQUENCES["validate"] + TARGET_SEQUENCES["test"] + TARGET_SEQUENCES["train"])
    processed_count = 0

    with zipfile.ZipFile(zip_path, 'r') as z:
        namelist = z.namelist()
        for name in namelist:
            # 仅处理包含 camera 且为 png 的文件
            if not ("camera" in name and name.endswith(".png")):
                continue

            parts = name.replace("\\", "/").split("/")
            # 判断路径中是否包含目标序列名
            matched_seq = None
            for p in parts:
                if p in all_target_seqs:
                    matched_seq = p
                    break
            if not matched_seq:
                continue

            # 寻找车辆 ID (通常为序列名之后的一级目录)
            seq_idx = parts.index(matched_seq)
            if seq_idx + 1 >= len(parts):
                continue
            veh_id = parts[seq_idx + 1]
            fname = parts[-1]
            jpg_name = os.path.splitext(fname)[0] + ".jpg"

            # 确定所属划分
            split_tag = "validate" if matched_seq in TARGET_SEQUENCES["validate"] else (
                "test" if matched_seq in TARGET_SEQUENCES["test"] else "train"
            )

            out_dir = os.path.join(output_base_dir, split_tag, matched_seq, veh_id)
            os.makedirs(out_dir, exist_ok=True)
            out_file = os.path.join(out_dir, jpg_name)

            if os.path.exists(out_file):
                continue

            # 从内存直接解码转换，不写临时 PNG
            with z.open(name) as f:
                with Image.open(f) as img:
                    img = img.convert("RGB")
                    img.save(out_file, format="JPEG", quality=quality, optimize=True)
            processed_count += 1
            if processed_count % 500 == 0:
                print(f"  - 已流式写入 {processed_count} 张 Q95 JPG...", flush=True)

    print(f"[+] 压缩包处理完成，累计高效提取 {processed_count} 张图片！", flush=True)


def download_and_extract(url_or_id, zip_name, output_base_dir, is_gdrive_id=True):
    tmp_zip = os.path.join("/root/autodl-tmp", zip_name)
    free_gb = check_disk_space()
    print(f"\n[Disk Check] 当前磁盘剩余: {free_gb:.1f} GB. 准备下载: {zip_name} ...", flush=True)

    # 1. 下载单分卷 (使用正确的 gdown 绝对路径)
    gdown_bin = "/root/.local/bin/gdown" if os.path.exists("/root/.local/bin/gdown") else "gdown"
    if is_gdrive_id:
        cmd = f"{gdown_bin} --fuzzy -O {tmp_zip} {url_or_id}"
    else:
        cmd = f"wget -c -O {tmp_zip} '{url_or_id}'"

    ret = subprocess.call(cmd, shell=True)
    if ret != 0 or not os.path.exists(tmp_zip):
        print(f"[!] 下载失败: {zip_name}，请检查网络或学术加速代理状态！", flush=True)
        return False

    # 2. 内存解压提取目标 Q95 JPG
    process_zip_to_q95(tmp_zip, output_base_dir)

    # 3. 立即彻底删除原压缩包！
    if os.path.exists(tmp_zip):
        os.remove(tmp_zip)
        print(f"[Cleanup] 已立即删除原始大压缩包 {zip_name}，释放磁盘空间！", flush=True)
    return True


def link_images_to_dataset(output_base_dir, links_dir="/root/autodl-tmp/proj_dir2_hetero_trust/data_links"):
    """将转换好的 JPG 建立软链接或就地挂载至现有训练环境"""
    print("[*] 正在同步图像软链接至实验环境...", flush=True)
    for sp, alias in [("validate", "opv2v_val"), ("test", "opv2v_test"), ("train", "opv2v_train")]:
        sp_src = os.path.join(output_base_dir, sp)
        if not os.path.exists(sp_src):
            continue
        link_target = os.path.join(links_dir, alias)
        # 遍历生成的序列建立局部软链接
        for seq in os.listdir(sp_src):
            seq_src = os.path.join(sp_src, seq)
            seq_target = os.path.join(link_target, seq)
            if not os.path.exists(seq_target):
                continue
            for veh in os.listdir(seq_src):
                veh_src = os.path.join(seq_src, veh)
                veh_target = os.path.join(seq_target, veh)
                if not os.path.exists(veh_target):
                    continue
                for img in os.listdir(veh_src):
                    src_f = os.path.join(veh_src, img)
                    dst_f = os.path.join(veh_target, img)
                    if not os.path.exists(dst_f):
                        try:
                            os.symlink(src_f, dst_f)
                        except Exception:
                            pass
    print("[+] 软链接同步完成，后续模型可直接透明访问 camera0~3.jpg！", flush=True)


def main():
    parser = argparse.ArgumentParser(description="服务端安全流式下载提取工具")
    parser.add_argument("--val_only", action="store_true", help="仅下载提取验证集 (快速就位)")
    parser.add_argument("--all", action="store_true", help="下载提取全部序列")
    parser.add_argument("--output_dir", default="/root/autodl-tmp/proj_dir2_hetero_trust/data_images_q95")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    check_disk_space()

    # 优先验证集
    print("=== 开始执行流式控盘提取 ===", flush=True)
    download_and_extract(GDRIVE_FILE_IDS["validate.zip"], "validate.zip", args.output_dir, is_gdrive_id=True)

    if args.all:
        download_and_extract(GDRIVE_FILE_IDS["test.zip"], "test.zip", args.output_dir, is_gdrive_id=True)

    link_images_to_dataset(args.output_dir)
    print("\n[SUCCESS] 图像流式提取完毕！磁盘安全受控。", flush=True)


if __name__ == "__main__":
    main()
