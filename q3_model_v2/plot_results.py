"""Static scientific figures from completed, fully paired Q3 holdout results.

Run with D:/Python/python.exe -X utf8 q3_model_v2/plot_results.py.
This script never imports a solver or calls an environment. Source truth is
used only to illustrate the already completed median-savings trajectory case.
"""
from __future__ import annotations

import argparse
import hashlib
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
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np


SCHEDULES = ("baseline", "v1", "lean")
LABELS = {"baseline": "原始初版", "v1": "上一版改进", "lean": "本次改进模型"}
VERSION_COLORS = {"baseline": "#99A6B0", "v1": "#668B96", "lean": "#3E726C"}
TEXT, MUTED = "#29383D", "#6A777C"
FIELD_STYLE = {
    "smooth": ("平滑误差场", "#527F89", "o"),
    "hash": ("位置哈希误差场", "#A08063", "s"),
    "extreme": ("边界跳变误差场", "#817691", "^"),
}
PARTS = (
    ("move", "移动", "#63878F"),
    ("measure", "检测", "#B0C6BD"),
    ("switch", "切频", "#D4BA86"),
    ("clear_success", "成功清除", "#AB9BB8"),
    ("clear_fail", "失败清除", "#C68E7B"),
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_style():
    """Require the requested Chinese font rather than silently lose glyphs."""
    name = "Microsoft YaHei"
    try:
        font_path = font_manager.findfont(name, fallback_to_default=False)
    except ValueError:
        candidate = Path("C:/Windows/Fonts/msyh.ttc")
        if not candidate.exists():
            raise RuntimeError("Microsoft YaHei is required for the Chinese figures")
        font_manager.fontManager.addfont(str(candidate))
        font_path = font_manager.findfont(name, fallback_to_default=False)
    plt.rcParams.update({
        "font.family": name, "font.size": 11,
        "axes.labelcolor": TEXT, "text.color": TEXT,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#C8D0D2", "axes.spines.top": False,
        "axes.spines.right": False, "axes.unicode_minus": False,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "svg.fonttype": "none",
    })
    return dict(family=name, file=str(font_path))


def validate_results(rows, summary, results):
    """Reject partial or mismatched aggregates; do not drop adverse cases."""
    groups = {s: {} for s in SCHEDULES}
    for row in rows:
        schedule = row["schedule"]
        if schedule not in groups:
            raise ValueError(f"Unexpected holdout schedule: {schedule}")
        key = (int(row["seed"]), row["field"])
        if key in groups[schedule]:
            raise ValueError(f"Duplicated condition: {schedule}, {key}")
        total = float(row["total_s"])
        parts = [float(row["time_parts_s"].get(part, 0.)) for part, _, _ in PARTS]
        if not math.isfinite(total) or total <= 0 or not np.isfinite(parts).all() or min(parts) < 0:
            raise ValueError(f"Invalid duration: {schedule}, {key}")
        if not math.isclose(sum(parts), total, abs_tol=1e-6):
            raise ValueError(f"Time components disagree with total: {schedule}, {key}")
        if not row["all_cleared"] or not row["certificate_valid"] or row.get("error"):
            raise ValueError(f"Cannot treat an incomplete run as a speed result: {schedule}, {key}")
        groups[schedule][key] = row
    keys = sorted(groups["baseline"])
    if not keys or any(set(group) != set(keys) for group in groups.values()):
        raise ValueError("All three versions must contain exactly the same completed conditions")
    for schedule, group in groups.items():
        saved = summary["groups"][schedule]
        mean = float(np.mean([row["total_s"] for row in group.values()]))
        if saved["runs"] != len(keys) or not math.isclose(saved["mean_total_s"], mean, abs_tol=1e-6):
            raise ValueError(f"Rows and holdout summary disagree: {schedule}")
    manifests = {}
    for field in sorted({key[1] for key in keys}):
        path = results / f"holdout-{field}" / "manifest.json"
        manifest = read_json(path)
        expected = {(seed, field) for seed in range(manifest["first_seed"], manifest["last_seed"] + 1)}
        actual = {key for key in keys if key[1] == field}
        if expected != actual or set(manifest["schedules"]) != set(SCHEDULES):
            raise ValueError(f"Holdout manifest is not fully represented: {field}")
        if manifest["split"] != "holdout" or manifest.get("official_simulator") is not False:
            raise ValueError("This figure requires completed local holdout runs")
        manifests[field] = dict(path=str(path), sha256=sha256(path))
    return keys, groups, manifests


def save_figure(fig, out, name):
    paths = []
    for suffix in ("png", "svg"):
        path = out / f"{name}.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=.16)
        paths.append(str(path))
    plt.close(fig)
    return paths


def plot_overview(keys, groups, out):
    old = np.array([groups["v1"][key]["total_s"] for key in keys])
    new = np.array([groups["lean"][key]["total_s"] for key in keys])
    original = np.array([groups["baseline"][key]["total_s"] for key in keys])
    savings = old - new
    reduction = float(100 * savings.sum() / old.sum())
    reduction_original = float(100 * (original.sum() - new.sum()) / original.sum())
    faster = int(np.sum(savings > 1e-6))
    slower = int(np.sum(savings < -1e-6))
    equal = len(keys) - faster - slower
    layouts = len({key[0] for key in keys})
    fields = sorted({key[1] for key in keys})

    fig = plt.figure(figsize=(13.8, 7.7))
    fig.text(.055, .952, "第三问：新模型的独立留出验证", fontsize=20, weight="bold")
    fig.text(.055, .900,
             f"{layouts} 种布局 × {len(fields)} 种固定地点误差场 · {len(keys)} 组相同条件 · 三个版本均全部清除 {len(keys)}/{len(keys)}",
             fontsize=11.5, color=MUTED)
    scatter = fig.add_axes([.075, .22, .375, .60])
    bars = fig.add_axes([.575, .345, .355, .35])

    low = math.floor(min(old.min(), new.min()) / 60 / 5) * 5 - 2
    high = math.ceil(max(old.max(), new.max()) / 60 / 5) * 5 + 2
    scatter.plot([low, high], [low, high], color="#AFB8BE", lw=1.2,
                 ls=(0, (4, 3)), zorder=1)
    for field in fields:
        selected = np.array([key[1] == field for key in keys])
        label, color, marker = FIELD_STYLE.get(field, (field, VERSION_COLORS["lean"], "o"))
        scatter.scatter(old[selected] / 60, new[selected] / 60, s=43, c=color,
                        marker=marker, alpha=.86, linewidths=.6,
                        edgecolors="white", label=label, zorder=3)
    scatter.set(xlim=(low, high), ylim=(low, high),
                xlabel="上一版总时间 / 分钟", ylabel="本次模型总时间 / 分钟")
    scatter.set_aspect("equal", adjustable="box")
    scatter.grid(color="#E8ECEE", lw=.65)
    scatter.set_title("每个点是一组配对条件", loc="left", fontsize=12, pad=13)
    scatter.text(.035, .965, "虚线下方：本次模型更快", transform=scatter.transAxes,
                 va="top", fontsize=10, color=MUTED)
    scatter.text(.035, .045, f"更快 {faster} 组 · 更慢 {slower} 组 · 持平 {equal} 组",
                 transform=scatter.transAxes, fontsize=10.5,
                 bbox=dict(facecolor="white", edgecolor="none", alpha=.94, pad=3))
    scatter.legend(loc="upper center", bbox_to_anchor=(.5, -.15), ncol=3,
                   frameon=False, fontsize=9, handletextpad=.35, columnspacing=1.2)

    accumulated = np.zeros(3)
    means = {}
    for key, label, color in PARTS:
        values = np.array([np.mean([row["time_parts_s"].get(key, 0.) for row in groups[s].values()])
                           for s in SCHEDULES]) / 60
        means[key] = {schedule: float(values[i] * 60) for i, schedule in enumerate(SCHEDULES)}
        bars.barh([2, 1, 0], values, left=accumulated, height=.47,
                  color=color, edgecolor="white", linewidth=.65, label=label)
        accumulated += values
    for y, total in zip([2, 1, 0], accumulated):
        bars.text(total + .7, y, f"{total:.2f}", va="center", fontsize=11, weight="bold")
    bars.set_yticks([2, 1, 0], [LABELS[s] for s in SCHEDULES])
    bars.set(xlim=(0, max(accumulated) * 1.15), ylim=(-.58, 2.58),
             xlabel="平均每局虚拟时间 / 分钟")
    bars.grid(axis="x", color="#E8ECEE", lw=.65)
    bars.set_axisbelow(True)
    bars.spines["left"].set_visible(False)
    bars.tick_params(axis="y", length=0)
    bars.legend(loc="upper left", bbox_to_anchor=(-.025, -.24), ncol=3,
                frameon=False, fontsize=9, handlelength=1.3, columnspacing=1.3)
    fig.text(.575, .835, "三版平均耗时组成", fontsize=12)
    fig.text(.575, .780, f"相对上一版降低 {reduction:.2f}%", fontsize=18,
             weight="bold", color=VERSION_COLORS["lean"])
    fig.text(.575, .733, f"每局平均节省 {savings.mean()/60:.2f} 分钟", fontsize=11.5, color=MUTED)
    fig.text(.575, .150, f"相对原始初版降低 {reduction_original:.2f}%", fontsize=10.5, color=MUTED)
    fig.text(.055, .045,
             "自建全向源环境；所有留出条件均保留。切频、检测、成功清除和失败清除均计费，图中包含移动时间。",
             fontsize=9.5, color=MUTED)
    outputs = save_figure(fig, out, "holdout_comparison")
    return dict(outputs=outputs, conditions=len(keys), independent_layouts=layouts,
                faster=faster, slower=slower, equal=equal,
                mean_total_s={s: float(np.mean([r["total_s"] for r in groups[s].values()])) for s in SCHEDULES},
                mean_components_s=means, reduction_vs_v1_pct=reduction,
                reduction_vs_original_pct=reduction_original, mean_saved_vs_v1_s=float(savings.mean()),
                includes_every_holdout_condition=True)


def plot_trajectories(keys, groups, results, out):
    reductions = np.array([100 * (groups["v1"][key]["total_s"] - groups["lean"][key]["total_s"])
                           / groups["v1"][key]["total_s"] for key in keys])
    median = float(np.median(reductions))
    index = min(range(len(keys)), key=lambda i: (abs(float(reductions[i]) - median), keys[i]))
    seed, field = keys[index]
    paths = [results / f"holdout-{field}" / "cases" / f"{seed}-{field}-{s}.json" for s in SCHEDULES]
    cases = [read_json(path) for path in paths]
    for schedule, case in zip(SCHEDULES, cases):
        if case["sources"] != cases[0]["sources"]:
            raise ValueError("Selected trajectories do not share identical sources")
        if not math.isclose(case["result"]["total_s"], groups[schedule][keys[index]]["total_s"], abs_tol=1e-6):
            raise ValueError("Selected case and holdout aggregate disagree")
    routes = [np.array([[0., 0.]] + [e["position"] for e in case["events"]]) for case in cases]
    sources = np.array([source["position"] for source in cases[0]["sources"]])
    angles = np.arange(6) * np.pi / 3
    stations = np.vstack(([0., 0.], 1200 * np.column_stack((np.cos(angles), np.sin(angles)))))
    extent = max(2050., math.ceil(max(float(np.abs(route).max()) for route in routes) / 250) * 250 + 120)

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 7.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=.065, right=.978, bottom=.205, top=.765, wspace=.16)
    fig.text(.047, .952, "第三问：中位收益案例的三版完整路线", fontsize=20, weight="bold")
    fig.text(.047, .898,
             f"按节省率最接近总体中位数选例 · 种子 {seed} · {FIELD_STYLE.get(field, (field,))[0]} · 本例相对上一版节省 {reductions[index]:.2f}%",
             fontsize=11.5, color=MUTED)
    for ax, schedule, case, route in zip(axes, SCHEDULES, cases, routes):
        color = VERSION_COLORS[schedule]
        ax.add_patch(Circle((0., 0.), 1800., fill=False, color="#AEBABC",
                            lw=1, ls=(0, (4, 3)), zorder=1))
        ax.plot(route[:, 0], route[:, 1], color=color, lw=1.25, alpha=.92, zorder=2)
        ax.scatter(stations[:, 0], stations[:, 1], marker="s", s=45, facecolors="white",
                   edgecolors="#87969B", linewidths=1.05, zorder=3)
        ax.scatter(sources[:, 0], sources[:, 1], marker="x", s=36,
                   color="#B56F53", linewidths=1.45, zorder=4)
        ax.scatter([0.], [0.], marker="*", s=135, color=TEXT,
                   edgecolors="white", linewidths=.35, zorder=5)
        ax.scatter(route[-1, 0], route[-1, 1], s=36, color=color,
                   edgecolors="white", linewidths=.8, zorder=5)
        segments = np.diff(route, axis=0)
        moving = np.flatnonzero(np.linalg.norm(segments, axis=1) > 150)
        if len(moving):
            for j in moving[np.linspace(0, len(moving) - 1, min(6, len(moving)), dtype=int)]:
                ax.annotate("", xy=route[j] + .60 * segments[j], xytext=route[j] + .47 * segments[j],
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.1, mutation_scale=9), zorder=3)
        ax.set_aspect("equal", adjustable="box")
        ax.set(xlim=(-extent, extent), ylim=(-extent, extent), xlabel="东向坐标 / 米")
        ax.set_xticks([-1800, -900, 0, 900, 1800])
        ax.set_yticks([-1800, -900, 0, 900, 1800])
        ax.tick_params(labelsize=9.5)
        ax.grid(color="#EDF0F1", lw=.65)
        row = case["result"]
        ax.set_title(f"{LABELS[schedule]}   {row['total_s']/60:.2f} 分钟\n"
                     f"移动 {row['distance_m']/1000:.2f} km · 清除 {row['cleared']}/{row['source_count']}",
                     fontsize=12, loc="left", pad=12)
    axes[0].set_ylabel("北向坐标 / 米")
    handles = [
        Line2D([], [], color="#668B96", lw=1.4, label="实际移动路线"),
        Line2D([], [], color="#87969B", marker="s", markerfacecolor="white", lw=0, markersize=6, label="固定覆盖参考点"),
        Line2D([], [], color="#B56F53", marker="x", lw=0, markersize=7, label="源真值（仅事后）"),
        Line2D([], [], color=TEXT, marker="*", lw=0, markersize=10, label="出发点"),
        Line2D([], [], color="#3E726C", marker="o", lw=0, markersize=6, label="结束位置"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.515, .094),
               ncol=5, frameon=False, fontsize=10, handlelength=1.6, columnspacing=1.5)
    fig.text(.047, .035,
             f"总体节省率中位数 {median:.2f}%；按绝对差最小选例，平局依次按种子、误差场排序。三图坐标范围一致；参考点不代表全部实际访问。",
             fontsize=9.5, color=MUTED)
    outputs = save_figure(fig, out, "median_trajectory_comparison")
    return dict(outputs=outputs, selected_seed=seed, selected_field=field,
                selection_rule="Nearest to median of per-condition percentage savings from v1 to lean; ties by seed, field.",
                condition_count=len(keys), median_saving_pct=median,
                selected_saving_pct=float(reductions[index]),
                distance_to_median_pct=abs(float(reductions[index]) - median),
                source_truth_used_only_for_post_run_plot=True,
                case_paths=[str(path) for path in paths], case_sha256=[sha256(path) for path in paths],
                total_s={s: groups[s][keys[index]]["total_s"] for s in SCHEDULES},
                common_axis_extent_m=extent)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    results, out = args.results.resolve(), args.out.resolve()
    rows_path, summary_path = results / "holdout_rows.json", results / "holdout_summary.json"
    rows, summary = read_json(rows_path), read_json(summary_path)
    keys, groups, manifests = validate_results(rows, summary, results)
    out.mkdir(parents=True, exist_ok=True)
    metadata = dict(local_only=True, data_files=[str(rows_path), str(summary_path)],
                    data_sha256={"rows": sha256(rows_path), "summary": sha256(summary_path)},
                    manifests=manifests, font=configure_style(),
                    schedule_labels=LABELS, overview=plot_overview(keys, groups, out),
                    median_trajectory=plot_trajectories(keys, groups, results, out))
    metadata["outputs_sha256"] = {str(Path(p).name): sha256(p)
        for section in (metadata["overview"], metadata["median_trajectory"]) for p in section["outputs"]}
    metadata["visual_qa"] = "pending actual image inspection"
    (out / "figure_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
