# -*- coding: utf-8 -*-
"""并行跑批、断点续跑与失败处理。"""
from __future__ import annotations

import csv
import os

import pybamm

from .model import build_model, run_one

_WORKER = {}


def _init_worker(cfg, baselines, parameter_set):
    pybamm.set_logging_level("ERROR")
    _WORKER["model"] = build_model()
    _WORKER["cfg"] = cfg
    _WORKER["baselines"] = baselines
    _WORKER["pset"] = parameter_set


def _eval_worker(args):
    idx, x, scenario_name = args
    sc = _WORKER["cfg"]["scenarios"][scenario_name]
    try:
        r = run_one(
            x, sc, _WORKER["baselines"], _WORKER["cfg"]["perturbation"],
            model=_WORKER["model"], parameter_set=_WORKER["pset"],
        )
        return idx, True, r, ""
    except Exception as e:  # 单次失败不影响整体
        return idx, False, {}, f"{type(e).__name__}: {e}"


def _result_path(checkpoint_dir, method, scenario_name):
    return os.path.join(checkpoint_dir, f"run_{method}_{scenario_name}.csv")


def _load_done(path):
    if not os.path.exists(path):
        return set(), None
    done = set()
    header = None
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        for row in reader:
            done.add(int(row["index"]))
    return done, header


def run_design(method, scenario_name, X, cfg, baselines, parameter_set="Chen2020",
               log_every=100):
    """
    对一个 (method, scenario) 跑全部样本，支持断点续跑。

    Returns
    -------
    (out_path, n_total, n_success, n_fail)
    """
    ckpt = cfg["runtime"]["checkpoint_dir"]
    os.makedirs(ckpt, exist_ok=True)
    path = _result_path(ckpt, method, scenario_name)
    out_cols = list(cfg["scenarios"][scenario_name]["outputs"].keys())
    header = ["index", "status"] + out_cols + ["error"]

    done, _ = _load_done(path)
    todo = [(i, X[i], scenario_name) for i in range(len(X)) if i not in done]

    new_file = not os.path.exists(path)
    fout = open(path, "a", encoding="utf-8-sig", newline="")
    writer = csv.writer(fout)
    if new_file:
        writer.writerow(header)

    n_success = n_fail = 0
    try:
        if todo:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            n_workers = cfg["runtime"]["n_workers"]
            with ProcessPoolExecutor(
                max_workers=n_workers,
                initializer=_init_worker,
                initargs=(cfg, baselines, parameter_set),
            ) as ex:
                futures = [ex.submit(_eval_worker, job) for job in todo]
                for k, fut in enumerate(as_completed(futures), 1):
                    idx, ok, r, err = fut.result()
                    if ok:
                        n_success += 1
                        row = [idx, "ok"] + [r[c] for c in out_cols] + [""]
                    else:
                        n_fail += 1
                        row = [idx, "fail"] + ["" for _ in out_cols] + [err]
                    writer.writerow(row)
                    if k % log_every == 0:
                        fout.flush()
                        print(f"  [{method}/{scenario_name}] {k}/{len(todo)} "
                              f"成功={n_success} 失败={n_fail}", flush=True)
    finally:
        fout.close()

    # 全量统计（含历史已完成）
    done_all, _ = _load_done(path)
    total = len(done_all)
    n_fail_total = 0
    n_succ_total = 0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "ok":
                n_succ_total += 1
            else:
                n_fail_total += 1
    return path, total, n_succ_total, n_fail_total
