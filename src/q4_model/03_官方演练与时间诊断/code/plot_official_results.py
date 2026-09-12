"""Render the five supplied official Q4 records without contacting a simulator."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import StrMethodFormatter


HERE = Path(__file__).resolve().parent
INPUT = HERE / "results" / "official_q4_20260911_analysis.json"
OUTPUT = HERE / "figures" / "official_q4_practice_costs"


def main() -> None:
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    cases = data["cases"]
    assert len(cases) == 5
    assert all(c["all_cleared_against_user_total"] for c in cases)
    cleared = sum(c["cleared"] for c in cases)
    total_sources = sum(c["total_sources_from_user"] for c in cases)
    assert cleared == total_sources == 63
    total_times = np.array([c["total_s"] for c in cases])
    means = np.array([c["average_s"] for c in cases])
    pooled = float(total_times.sum() / total_sources)

    plt.rcParams.update(
        {
            "font.family": "Microsoft YaHei",
            "axes.unicode_minus": False,
            "font.size": 11,
            "axes.labelcolor": "#46505B",
            "text.color": "#25323E",
            "xtick.color": "#46505B",
            "ytick.color": "#46505B",
            "svg.fonttype": "path",
            "savefig.facecolor": "#FAFBFC",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.3), facecolor="#FAFBFC")
    fig.subplots_adjust(left=0.068, right=0.975, top=0.73, bottom=0.20, wspace=0.25)
    fig.text(
        0.068,
        0.933,
        f"第四问｜五次官方记录全部清除（{cleared} / {total_sources}）",
        fontsize=21,
        fontweight="bold",
    )
    fig.text(
        0.068,
        0.875,
        "清除数由操作反馈核对；源总数及全向、定向组成由用户提供。",
        color="#65717C",
        fontsize=11,
    )

    for ax in axes:
        ax.set_facecolor("#FAFBFC")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#CBD2D7")
        ax.grid(axis="y", color="#E4E8EC", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, pad=8)
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    x = np.arange(5)
    bottom = np.zeros(5)
    components = [
        ("move", "移动", "#8096AB"),
        ("measure", "检测", "#ADC0B7"),
        ("switch", "切频", "#D2BD8D"),
        ("clear_success", "成功清除", "#6F9784"),
        ("clear_fail", "失败清除", "#BD8983"),
    ]
    for key, label, color in components:
        values = np.array([c["costs_s"][key] for c in cases])
        axes[0].bar(x, values, bottom=bottom, color=color, width=0.61, label=label)
        bottom += values
    assert np.allclose(bottom, total_times, rtol=0, atol=0.0001)
    for i, t in enumerate(total_times):
        axes[0].text(i, t + 130, f"{t:,.0f}", ha="center", fontsize=11)
    axes[0].set_title("每场总时间及成本构成", loc="left", fontsize=14, pad=22)
    axes[0].set_ylabel("虚拟秒", labelpad=10)
    axes[0].set_ylim(0, 8100)
    axes[0].set_xticks(x, [f"第{i + 1}场" for i in x])
    axes[0].legend(
        loc="lower left",
        bbox_to_anchor=(-0.012, -0.235),
        ncol=5,
        frameon=False,
        fontsize=9.5,
        columnspacing=1.1,
        handlelength=1.2,
        handletextpad=0.5,
    )

    axes[1].bar(x, means, color="#879FA9", width=0.61)
    axes[1].axhline(pooled, color="#7C6747", linewidth=1.2, linestyle=(0, (4, 3)))
    axes[1].text(
        0.97,
        0.975,
        f"合计单源时间  {pooled:.1f} 秒/源",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        color="#7C6747",
        fontsize=10.5,
    )
    for i, value in enumerate(means):
        axes[1].text(i, value + 11, f"{value:.1f}", ha="center", fontsize=11)
    axes[1].set_title("每场单源时间与源类型组成", loc="left", fontsize=14, pad=22)
    axes[1].set_ylabel("虚拟秒 / 源", labelpad=10)
    axes[1].set_ylim(0, 650)
    axes[1].set_xticks(
        x,
        [
            f"第{i + 1}场\n全向 {c['omni_from_user']}\n定向 {c['directional_from_user']}"
            for i, c in enumerate(cases)
        ],
    )
    axes[1].tick_params(axis="x", labelsize=10)
    fig.text(
        0.068,
        0.065,
        "全部时间均为题目计费的虚拟时间。单源时间 = 该场总时间 ÷ 源数；虚线 = 五场总时间 ÷ 63。",
        fontsize=10,
        color="#65717C",
    )
    fig.text(
        0.068,
        0.025,
        "各场布局及源数不同；本图用于描述这五场表现，不能单独推断源类型对耗时的因果影响。",
        fontsize=10,
        color="#65717C",
    )
    OUTPUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=180)
    fig.savefig(OUTPUT.with_suffix(".svg"))
    plt.close(fig)
    print(f"Saved {OUTPUT.with_suffix('.png')}")
    print(f"Saved {OUTPUT.with_suffix('.svg')}")
    print(f"Verified {cleared}/{total_sources}; pooled virtual time {pooled:.6f} s/source")


if __name__ == "__main__":
    main()
