# -*- coding: utf-8 -*-
"""
P2D GSA 平台入口（向后兼容薄包装）。

完整实现已移入包内模块 `p2d_gsa.cli`。pip 安装后可直接使用 `p2d-gsa` 命令；
本文件保留以支持直接 `python run_gsa.py ...` 的既有用法。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from p2d_gsa.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
