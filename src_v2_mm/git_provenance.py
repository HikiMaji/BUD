# -*- coding: utf-8 -*-
"""
实验代码版本溯源与自动提交辅助模块（Git Provenance Helper）。

功能：
1. 检查工作区是否有未提交的已追踪代码/配置改动（保障每次实验对应唯一确定 commit）；
2. 读取当前 git commit hash / short hash / branch；
3. 将当前 git 提交号写入结果 JSON 的 meta.git_commit 字段；
4. 实验步骤完成后仅暂存代码、配置与结果 JSON 并执行自动提交（不提交日志、权重、缓存、图像）。
"""

import argparse
import json
import os
import subprocess
import sys
from typing import List, Optional

DEFAULT_REPO = "/root/autodl-tmp/proj_dir2_hetero_trust"


def _run_git(args: List[str], repo_dir: str = DEFAULT_REPO) -> str:
    res = subprocess.run(
        ["git"] + args,
        cwd=repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        check=True,
    )
    return res.stdout.strip()


def get_git_commit_info(repo_dir: str = DEFAULT_REPO) -> dict:
    """返回当前仓库 commit hash、短 hash、分支名与工作区脏状态。"""
    commit = _run_git(["rev-parse", "HEAD"], repo_dir)
    short = _run_git(["rev-parse", "--short", "HEAD"], repo_dir)
    branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], repo_dir)
    subject = _run_git(["log", "-1", "--pretty=%s"], repo_dir)
    status = _run_git(["status", "--porcelain", "--untracked-files=no"], repo_dir)
    return {
        "git_commit": commit,
        "git_short": short,
        "git_branch": branch,
        "git_subject": subject,
        "git_dirty": bool(len(status) > 0),
    }


def assert_clean_worktree(repo_dir: str = DEFAULT_REPO, allow_logs_dirty: bool = True) -> dict:
    """
    检查工作区是否无未提交的代码/配置改动。
    allow_logs_dirty=True 时仅忽略 logs/ 下正在实时写入的运行日志文件。
    """
    raw = _run_git(["status", "--porcelain"], repo_dir)
    dirty_items = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if allow_logs_dirty and (path.startswith("logs/") or path.endswith(".log")):
            continue
        dirty_items.append(line)
    if dirty_items:
        joined = "\n  ".join(dirty_items)
        raise RuntimeError(f"工作区存在未提交的改动，拒绝启动实验步骤：\n  {joined}")
    return get_git_commit_info(repo_dir)


def inject_git_commit_to_json(json_path: str, repo_dir: str = DEFAULT_REPO, commit_info: Optional[dict] = None) -> dict:
    """将当前 git 提交号写入结果 JSON 的 meta 字段并原子回写。"""
    info = commit_info or get_git_commit_info(repo_dir)
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "meta" not in obj or not isinstance(obj["meta"], dict):
        obj["meta"] = {}
    obj["meta"]["git_commit"] = info["git_commit"]
    obj["meta"]["git_short"] = info["git_short"]
    obj["meta"]["git_branch"] = info["git_branch"]
    obj["meta"]["git_dirty_at_write"] = info["git_dirty"]
    tmp = json_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, json_path)
    return info


def auto_commit_code_and_results(
    commit_msg: str,
    result_paths: List[str],
    repo_dir: str = DEFAULT_REPO,
) -> Optional[str]:
    """
    步骤跑完后自动提交：仅提交代码（src_v2_mm/ scripts_v2_mm/ .gitignore）与指定结果 JSON，
    不自动提交临时日志、权重、图像或缓存。
    """
    targets = [".gitignore", "src_v2_mm", "scripts_v2_mm"]
    for rp in result_paths:
        if rp and os.path.exists(os.path.join(repo_dir, rp) if not os.path.isabs(rp) else rp):
            targets.append(rp)
    subprocess.run(["git", "add"] + targets, cwd=repo_dir, check=True)
    staged = _run_git(["diff", "--cached", "--name-only"], repo_dir)
    if not staged:
        print("[git_provenance] 无新增代码或结果 JSON 改动，跳过提交。")
        return None
    subprocess.run(["git", "commit", "-m", commit_msg], cwd=repo_dir, check=True)
    new_hash = _run_git(["rev-parse", "--short", "HEAD"], repo_dir)
    print(f"[git_provenance] 已自动提交 {new_hash}: {commit_msg}")
    return new_hash


def main():
    parser = argparse.ArgumentParser(description="Git 版本校验、JSON 注入与自动提交工具")
    parser.add_argument("--repo", type=str, default=DEFAULT_REPO)
    parser.add_argument("--check-clean", action="store_true")
    parser.add_argument("--inject-json", nargs="+", default=None)
    parser.add_argument("--auto-commit", type=str, default=None)
    parser.add_argument("--result-files", nargs="*", default=[])
    args = parser.parse_args()

    if args.check_clean:
        info = assert_clean_worktree(args.repo, allow_logs_dirty=True)
        print(f"[git_provenance] 工作区干净，当前基线: {info['git_short']} ({info['git_branch']}) - {info['git_subject']}")
    if args.inject_json:
        for jp in args.inject_json:
            info = inject_git_commit_to_json(jp, args.repo)
            print(f"[git_provenance] 已写入 commit {info['git_short']} -> {jp}")
    if args.auto_commit:
        auto_commit_code_and_results(args.auto_commit, args.result_files, args.repo)


if __name__ == "__main__":
    main()
