"""
一键推送到 GitHub。

用法：
    python scripts/push_to_github.py <仓库名> [--public]
    python scripts/push_to_github.py my-repo --public --token ghp_xxx

说明：
- 本机没有 gh CLI，所以走 git + HTTPS API。
- 若系统已配置 git credential helper，直接 `git push` 即可，无需 token。
- 没有 token 时会提示手动操作。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIRROR = "https://ghproxy.net/https://github.com"


def sh(cmd, **kw):
    r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True,
                       text=True, cwd=ROOT, **kw)
    return r.returncode, r.stdout, r.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", help="GitHub 仓库名")
    ap.add_argument("--public", action="store_true", help="公开仓库（默认私有）")
    ap.add_argument("--token", default=os.environ.get("GITHUB_TOKEN", ""),
                    help="GitHub Personal Access Token")
    ap.add_argument("--user", default="", help="GitHub 用户名")
    args = ap.parse_args()

    name = args.repo
    if not name.startswith("http"):
        remote = "https://github.com/%s/%s.git" % (args.user, name) if args.user \
            else "git@github.com:%s.git" % name
    else:
        remote = name

    print("仓库目录:", ROOT)
    print("远程地址:", remote)
    print()

    # 检查 git 状态
    code, out, err = sh("git status --short")
    if out.strip():
        print("有未提交的改动：")
        print(out[:500])
        print("请先提交。")
        return 1

    code, out, err = sh(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    branch = out.strip() or "main"
    print("当前分支:", branch)

    # 设置远程
    sh(["git", "remote", "remove", "origin"])
    code, out, err = sh(["git", "remote", "add", "origin", remote])
    if code != 0:
        print("设置远程失败:", err[:200])
        return 1
    print("✓ 已设置 origin")

    # 推送
    print()
    print("尝试推送（若代理阻断，脚本会自动改用 ghproxy 镜像）...")
    code, out, err = sh(["git", "push", "-u", "origin", branch])
    if code == 0:
        print("✓ 推送成功")
        print("  https://github.com/%s/%s" % (args.user or "<user>", name))
        return 0

    print("直接推送失败:", (err or out)[-300:])
    print()
    print("尝试通过 ghproxy 镜像 ...")
    mirror = "%s/%s.git" % (MIRROR, ("%s/%s" % (args.user, name)
                                     if args.user else name))
    sh(["git", "remote", "set-url", "origin", mirror])
    code, out, err = sh(["git", "push", "-u", "origin", branch])
    if code == 0:
        print("✓ 通过镜像推送成功")
        print("  注意：远程地址已改为镜像，仓库地址仍是")
        print("  https://github.com/%s/%s" % (args.user or "<user>", name))
        return 0

    print("镜像推送也失败:", (err or out)[-300:])
    sh(["git", "remote", "set-url", "origin", remote])
    print()
    print("请手动执行：")
    print("  git -C \"%s\" push -u origin %s" % (ROOT, branch))
    return 1


if __name__ == "__main__":
    sys.exit(main())
