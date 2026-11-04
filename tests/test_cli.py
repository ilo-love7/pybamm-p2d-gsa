# -*- coding: utf-8 -*-
"""CLI 入口单元测试：parser、main 直接调度（monkeypatch 隔离副作用）+ 薄包装。"""
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

import p2d_gsa.cli as cli
from p2d_gsa.cli import build_parser, cmd_sample
from p2d_gsa.config import load_config

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN_GSA = os.path.join(ROOT, "run_gsa.py")


def test_build_parser_subcommands():
    parser = build_parser()
    args = parser.parse_args(["sample"])
    assert args.command == "sample"
    args = parser.parse_args(["run", "--method", "morris",
                              "--scenario", "s1_discharge_1c"])
    assert args.method == "morris"
    assert args.scenario == "s1_discharge_1c"
    assert parser.parse_args(["analyze"]).command == "analyze"


def test_main_dispatches_each_command(monkeypatch):
    # 隔离配置加载与真实副作用，只验证 main 的解析与分发
    monkeypatch.setattr(cli, "load_config", lambda p: {"test": True})
    cases = [
        (["sample"], "cmd_sample"),
        (["analyze"], "cmd_analyze"),
        (["run", "--method", "morris", "--scenario", "s1"], "cmd_run"),
    ]
    for argv, cmd_attr in cases:
        hit = {}

        def fake(cfg, args, _hit=hit):
            _hit["cfg"] = cfg
            _hit["args"] = args

        monkeypatch.setattr(cli, cmd_attr, fake)
        monkeypatch.setattr(sys, "argv", ["p2d-gsa"] + argv)
        cli.main()
        assert hit["cfg"] == {"test": True}
        assert hit["args"] is not None


def test_cmd_sample_generates_designs(tmp_path):
    cfg = load_config()
    cfg["runtime"]["checkpoint_dir"] = str(tmp_path)
    import argparse
    args = argparse.Namespace()
    cmd_sample(cfg, args)
    assert (tmp_path / "design_morris.npy").exists()
    assert (tmp_path / "design_saltelli.npy").exists()


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
