"""Reproducible post-run figures for paired local Question 3 experiments.

Usage: D:/Python/python.exe -X utf8 q3_improved/plot_results.py
Only completed rows/summary files are read.  True source locations are used
exclusively for these after-the-run illustrations, never for the solver.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "tmp" / "matplotlib"
CONFIG.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(CONFIG)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle
import numpy as np


TEXT = "#29383D"
MUTED = "#6A777C"
BASE = "#778B9C"
DYNAMIC = "#426F70"
FIELD_STYLE = {
    "smooth": ("平滑误差场", "#527F89", "o"),
    "hash": ("位置哈希误差场", "#A08063", "s"),
    "extreme": ("边界跳变误差场", "#817691", "^"),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def configure_style():
    available = {font.name for font in font_manager.fontManager.ttflist}
    preferred = next((name for name in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC")
                      if name in available), "DejaVu Sans")
    plt.rcParams.update({
        "font.family": preferred,
        "font.size": 11,
        "axes.labelcolor": TEXT,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.edgecolor": "#C8D0D2",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
    })
    return preferred


def paired_rows(rows, summary):
    by_schedule = {}
    for row in rows:
        if row["schedule"] not in ("baseline", "dynamic"):
            continue
        key = (row["seed"], row["field"])
        group = by_schedule.setdefault(row["schedule"], {})
        if key in group:
            raise ValueError(f"Duplicate condition: {key}, {row['schedule']}")
        group[key] = row
    if not by_schedule.get("baseline") or not by_schedule.get("dynamic"):
        raise ValueError("Both baseline and dynamic completed results are required")
    if by_schedule["baseline"].keys() != by_schedule["dynamic"].keys():
        raise ValueError("Baseline and dynamic conditions are not fully paired")
    pairs = [(by_schedule["baseline"][key], by_schedule["dynamic"][key])
             for key in sorted(by_schedule["baseline"])]
    for schedule, group in by_schedule.items():
        actual = np.mean([row["total_s"] for row in group.values()])
        saved = summary["groups"][schedule]
        if len(group) != saved["runs"] or not math.isclose(actual, saved["mean_total_s"], abs_tol=1e-6):
            raise ValueError("rows.json and summary.json disagree; do not plot partial results")
    if any(base["total_s"] <= 0 or dynamic["total_s"] <= 0 for base, dynamic in pairs):
        raise ValueError("Nonpositive run duration cannot be used in paired comparisons")
    return pairs


def save_figure(fig, out, name):
    paths = []
    for suffix in ("png", "svg"):
        path = out / f"{name}.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=.15)
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_paired(pairs, summary, out):
    baseline = np.array([base["total_s"] for base, _ in pairs])
    dynamic = np.array([other["total_s"] for _, other in pairs])
    savings = baseline-dynamic
    reduction = float(100*savings.sum()/baseline.sum())
    faster = int(np.sum(savings > 1e-6))
    slower = int(np.sum(savings < -1e-6))
    equal = len(pairs)-faster-slower
    both_cleared = sum(base["all_cleared"] and other["all_cleared"] for base, other in pairs)
    layouts = len({base["seed"] for base, _ in pairs})
    fields = sorted({base["field"] for base, _ in pairs})

    fig = plt.figure(figsize=(13.2, 6.8))
    fig.text(.055, .955, "第三问：独立留出集的配对结果", fontsize=20, weight="bold")
    fig.text(.055, .901,
             f"{layouts} 种布局 × {len(fields)} 种固定地点误差场 · {len(pairs)} 组相同条件 · 两方案均全清 {both_cleared}/{len(pairs)}",
             fontsize=11.5, color=MUTED)
    left = fig.add_axes([.085, .20, .365, .61])
    right = fig.add_axes([.56, .34, .37, .37])

    low = math.floor(min(baseline.min(), dynamic.min())/60/5)*5-2
    high = math.ceil(max(baseline.max(), dynamic.max())/60/5)*5+2
    left.plot([low, high], [low, high], color="#AFB8BE", lw=1.2, ls=(0, (4, 3)), zorder=1)
    for field in fields:
        mask = np.array([base["field"] == field for base, _ in pairs])
        label, color, marker = FIELD_STYLE.get(field, (field, DYNAMIC, "o"))
        left.scatter(baseline[mask]/60, dynamic[mask]/60, c=color, marker=marker,
                     s=40, alpha=.84, linewidths=.6, edgecolors="white", label=label, zorder=3)
    left.set(xlim=(low, high), ylim=(low, high), xlabel="原版总时间 / 分钟", ylabel="动态调度总时间 / 分钟")
    left.set_aspect("equal", adjustable="box")
    left.grid(color="#E8ECEE", lw=.65, zorder=0)
    left.set_title("每个点是一组配对条件", loc="left", fontsize=12, pad=13)
    left.text(.035, .965, "虚线下方：动态调度更快", transform=left.transAxes,
              va="top", fontsize=10, color=MUTED)
    left.text(.035, .045, f"更快 {faster} 组 · 更慢 {slower} 组 · 持平 {equal} 组",
              transform=left.transAxes, fontsize=10.5,
              bbox=dict(facecolor="white", edgecolor="none", alpha=.92, pad=3))
    left.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=3,
                frameon=False, fontsize=9, handletextpad=.35, columnspacing=1.2)

    parts = [
        ("move", "移动", "#66878F"),
        ("measure", "检测", "#B0C6BD"),
        ("switch", "切频", "#D4BA86"),
        ("clear_success", "成功清除", "#AB9BB8"),
    ]
    if any(row["time_parts_s"].get("clear_fail", 0) > 0 for pair in pairs for row in pair):
        parts.append(("clear_fail", "失败清除", "#C68E7B"))
    accumulated = np.zeros(2)
    for key, label, color in parts:
        values = np.array([np.mean([pair[index]["time_parts_s"].get(key, 0) for pair in pairs])
                           for index in range(2)])/60
        right.barh([1, 0], values, left=accumulated, height=.42, color=color,
                   edgecolor="white", linewidth=.7, label=label)
        accumulated += values
    for y, total in zip([1, 0], accumulated):
        right.text(total+.7, y, f"{total:.2f}", va="center", fontsize=11, weight="bold")
    right.set_yticks([1, 0], ["原版", "动态调度"])
    right.set(xlim=(0, max(accumulated)*1.17), ylim=(-.6, 1.6), xlabel="平均每局虚拟时间 / 分钟")
    right.grid(axis="x", color="#E8ECEE", lw=.65)
    right.set_axisbelow(True)
    right.spines["left"].set_visible(False)
    right.tick_params(axis="y", length=0)
    fig.text(.56, .845, "平均耗时组成", fontsize=12)
    right.legend(loc="upper left", bbox_to_anchor=(-.05, -.26), ncol=4,
                 frameon=False, fontsize=9, handlelength=1.2, columnspacing=1.1)
    fig.text(.56, .785, f"平均总时间下降 {reduction:.2f}%", fontsize=18,
             weight="bold", color=DYNAMIC)
    fig.text(.56, .735, f"每局平均节省 {savings.mean()/60:.2f} 分钟", fontsize=11.5, color=MUTED)
    tail_base = summary["groups"]["baseline"]["mean_scan_tail_s"]/60
    tail_dynamic = summary["groups"]["dynamic"]["mean_scan_tail_s"]/60
    fig.text(.56, .17, f"最后一源清除后的平均剩余搜索：{tail_base:.2f} → {tail_dynamic:.2f} 分钟",
             fontsize=10, color=MUTED)
    fig.text(.055, .035,
             "数据来自自建全向源环境；同一布局在三个误差场中重复测试。所有点均保留，含动态方案更慢的条件。",
             fontsize=9.5, color=MUTED)
    outputs = save_figure(fig, out, "paired_results")
    return dict(outputs=outputs, conditions=len(pairs), independent_layouts=layouts,
                faster=faster, slower=slower, equal=equal, paired_all_cleared=both_cleared,
                mean_baseline_s=float(baseline.mean()), mean_dynamic_s=float(dynamic.mean()),
                aggregate_reduction_pct=reduction, mean_saved_s=float(savings.mean()))


def plot_trajectory(pairs, data_dir, out):
    reductions = np.array([100*(base["total_s"]-dynamic["total_s"])/base["total_s"]
                           for base, dynamic in pairs])
    median = float(np.median(reductions))
    index = min(range(len(pairs)), key=lambda i: (abs(float(reductions[i])-median),
                pairs[i][0]["seed"], pairs[i][0]["field"]))
    base_row, dynamic_row = pairs[index]
    seed, field = base_row["seed"], base_row["field"]
    paths = [data_dir / "cases" / f"{seed}-{field}-{schedule}.json"
             for schedule in ("baseline", "dynamic")]
    cases = [read_json(path) for path in paths]
    if cases[0]["sources"] != cases[1]["sources"]:
        raise ValueError("Selected trajectory cases do not share the same sources")
    for case, expected in zip(cases, pairs[index]):
        if not math.isclose(case["result"]["total_s"], expected["total_s"], abs_tol=1e-6):
            raise ValueError("Case file and rows.json disagree")

    angles = np.arange(6)*np.pi/3
    scans = np.vstack(([0., 0.], 1200*np.column_stack((np.cos(angles), np.sin(angles)))))
    positions = [np.array([[0., 0.]]+[event["position"] for event in case["events"]])
                 for case in cases]
    sources = np.array([source["position"] for source in cases[0]["sources"]])
    extent = max(2050., math.ceil(max(float(np.abs(p).max()) for p in positions)/250)*250+150)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 7.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.97, bottom=.19, top=.79, wspace=.17)
    fig.text(.055, .955, "第三问：典型配对案例的完整路线", fontsize=20, weight="bold")
    fig.text(.055, .901,
             f"按节省率最接近总体中位数选取 · 种子 {seed} · {FIELD_STYLE.get(field, (field,))[0]} · 本例节省 {reductions[index]:.2f}%",
             fontsize=11.5, color=MUTED)

    for ax, case, route, label, color in zip(axes, cases, positions,
                                           ("原版", "动态调度"), (BASE, DYNAMIC)):
        ax.add_patch(Circle((0, 0), 1800, fill=False, color="#AEBABC", lw=1, ls=(0, (4, 3)), zorder=1))
        ax.plot(route[:, 0], route[:, 1], color=color, lw=1.15, alpha=.85,
                label="实际移动路线", zorder=2)
        ax.scatter(scans[:, 0], scans[:, 1], marker="s", s=58, facecolors="white",
                   edgecolors="#6F8185", linewidths=1.15, label="七个覆盖点", zorder=3)
        ax.scatter(sources[:, 0], sources[:, 1], marker="x", s=38,
                   color="#B56F53", linewidths=1.5, label="源真值（仅事后）", zorder=4)
        ax.scatter([0], [0], marker="*", s=130, color=TEXT, linewidths=.3,
                   edgecolors="white", label="出发点", zorder=5)
        ax.scatter(route[-1, 0], route[-1, 1], marker="o", s=32, color=color,
                   edgecolors="white", linewidths=.7, label="结束位置", zorder=5)
        # A few arrows make the time direction visible without hiding close stops.
        segments = np.diff(route, axis=0)
        moving = np.flatnonzero(np.linalg.norm(segments, axis=1) > 150)
        if len(moving):
            for j in moving[np.linspace(0, len(moving)-1, min(5, len(moving)), dtype=int)]:
                mid = route[j] + .58*segments[j]
                before = route[j] + .48*segments[j]
                ax.annotate("", xy=mid, xytext=before,
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, mutation_scale=9), zorder=3)
        ax.set_aspect("equal", adjustable="box")
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), xlabel="东向坐标 / 米")
        ax.set_xticks([-1800, -900, 0, 900, 1800])
        ax.set_yticks([-1800, -900, 0, 900, 1800])
        ax.grid(color="#EDF0F1", lw=.65, zorder=0)
        result = case["result"]
        ax.set_title(f"{label}   {result['total_s']/60:.2f} 分钟\n"
                     f"移动 {result['distance_m']/1000:.2f} km · 清除 {result['cleared']}/{result['source_count']}",
                     fontsize=12, loc="left", pad=12)
    axes[0].set_ylabel("北向坐标 / 米")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.52, .078),
               ncol=5, frameon=False, fontsize=10, handlelength=1.6, columnspacing=1.5)
    fig.text(.055, .025,
             f"总体节省率中位数为 {median:.2f}%；按与中位数的绝对差最小选例，平局依次按种子、误差场排序。两图坐标范围完全一致。",
             fontsize=9.5, color=MUTED)
    outputs = save_figure(fig, out, "trajectory_comparison")
    return dict(outputs=outputs, selected_seed=seed, selected_field=field,
                selection="Minimum absolute distance of per-condition percentage saving to median across all baseline/dynamic pairs; ties by seed, field.",
                condition_count=len(pairs), median_reduction_pct=median,
                selected_reduction_pct=float(reductions[index]),
                distance_to_median_pct=abs(float(reductions[index])-median),
                source_truth_used_only_for_post_run_plot=True, case_paths=[str(path) for path in paths],
                total_baseline_s=base_row["total_s"], total_dynamic_s=dynamic_row["total_s"],
                common_axis_extent_m=extent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results" / "holdout-v1")
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    data_dir, out = args.results.resolve(), args.out.resolve()
    rows = read_json(data_dir / "rows.json")
    summary = read_json(data_dir / "summary.json")
    pairs = paired_rows(rows, summary)
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(local_only=True, results_directory=str(data_dir), font=configure_style(),
                    paired_results=plot_paired(pairs, summary, out),
                    trajectory_comparison=plot_trajectory(pairs, data_dir, out))
    path = out / "figure_metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))


if __name__ == "__main__":
    main()
