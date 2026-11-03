# -*- coding: utf-8 -*-
"""配置加载：显式路径 > 环境变量 P2D_GSA_CONFIG > 项目根 configs > 包内默认。"""
from __future__ import annotations

import os
from importlib.resources import files as _resource_files

import yaml

_DEFAULT_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "configs",
    "gsa_config.yaml",
)


def _resolve_config_ref(path=None):
    if path:
        return path
    env = os.environ.get("P2D_GSA_CONFIG")
    if env:
        return env
    if os.path.exists(_DEFAULT_CONFIG):
        return _DEFAULT_CONFIG
    return _resource_files("p2d_gsa.resources").joinpath("gsa_config.yaml")


def load_config(path=None):
    """加载 YAML 配置并返回 dict。"""
    ref = _resolve_config_ref(path)
    if isinstance(ref, str):
        with open(ref, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    with ref.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
