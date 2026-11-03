# -*- coding: utf-8 -*-
"""配置加载测试。"""
from p2d_gsa.config import load_config


def test_load_config_keys():
    cfg = load_config()
    assert {"model", "scenarios", "morris", "sobol", "runtime"}.issubset(cfg)
    assert len(cfg["scenarios"]) == 3
