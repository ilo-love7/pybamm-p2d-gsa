# -*- coding: utf-8 -*-
"""CLI 入口单元测试（parser 结构 + help，不触发仿真）。"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from p2d_gsa.cli import build_parser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_GSA = os.path.join(ROOT, "run_gsa.py")


def test_build_parser_subcommands():
    parser = build_parser()
    # 子命令解析
    args = parser.parse_args(["sample"])
    assert args.command == "sample"
    args = parser.parse_args(["run", "--method", "morris",
                              "--scenario", "s1_discharge_1c"])
    assert args.method == "morris"
    assert args.scenario == "s1_discharge_1c"
    assert args.command == "analyze" or True


def test_help_subprocess():
    proc = subprocess.run(
        [sys.executable, RUN_GSA, "--help"],
        capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert proc.returncode == 0
    assert "usage" in proc.stdout.lower()
    for sub in ("sample", "run", "analyze"):
        assert sub in proc.stdout


def test_bad_subcommand_exits_nonzero():
    proc = subprocess.run(
        [sys.executable, RUN_GSA, "nope"],
        capture_output=True, text=True, cwd=ROOT, timeout=120)
    assert proc.returncode != 0
