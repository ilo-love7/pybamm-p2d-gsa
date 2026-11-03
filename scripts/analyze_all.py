# -*- coding: utf-8 -*-
"""全量分析（向后兼容薄包装）：核心编排已移入 `p2d_gsa.pipeline`。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from p2d_gsa.config import load_config  # noqa: E402
from p2d_gsa.pipeline import run_analysis  # noqa: E402


def main():
    run_analysis(load_config())


if __name__ == "__main__":
    main()
