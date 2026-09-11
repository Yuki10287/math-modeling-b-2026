"""Plot the frozen paired cost-model experiment, including negative results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / 'results/cost_holdout_summary.json').read_text())
plt.rcParams.update({'font.sans-serif': ['Microsoft YaHei'], 'axes.unicode_minus': False,
    'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
labels = {'uniform': '普通分布（12组）', 'sparse': '10源稀疏（9组）', 'boundary': '边界分布（9组）'}
colors = {'uniform': '#607d8b', 'sparse': '#7c9279', 'boundary': '#b99370'}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), gridspec_kw={'width_ratios': [1.4, 1]})
for kind, label in labels.items():
    pairs = [p for p in data['pairs'] if p['population'] == kind]
    axes[0].scatter([p['original_s'] for p in pairs], [p['saved_s'] for p in pairs],
        label=label, color=colors[kind], s=35, alpha=.8, edgecolors='white', linewidths=.4)
axes[0].axhline(0, color='#555555', linewidth=.8)
axes[0].set(xlabel='上一版联合调度总时间 / 秒', ylabel='节省时间 / 秒（正值表示修正版更快）',
    title='新布局配对：9组更快，5组相同，16组更慢')
axes[0].grid(axis='y', alpha=.15)
axes[0].legend(frameon=False, fontsize=8, loc='lower right')
means = [data['populations'][k]['comparisons_to_original']['bounded']['mean_saved_s'] for k in labels]
axes[1].barh(list(labels.values())[::-1], means[::-1], color=list(colors.values())[::-1], height=.55)
axes[1].axvline(0, color='#555555', linewidth=.8)
axes[1].set(xlabel='平均节省时间 / 秒', title='三类布局均值均退步', xlim=(-30, 3))
for i, value in enumerate(means[::-1]):
    axes[1].text(value - .5, i, f'{value:+.2f}', va='center', ha='right')
fig.suptitle('第三问成本修正实验：3124.36 → 3136.17秒，平均增加0.38%', fontsize=14, y=.98)
fig.text(.02, .045, '10种新布局 × 3种误差场；比较对象为上一版联合调度候选。虚拟动作时间来自本地自建环境。',
    fontsize=9, color='#555555')
fig.text(.02, .01, '按布局分层重抽样的平均节省95%区间：−24.00～−2.38秒。未运行官方模拟器；本轮不采用修正版。',
    fontsize=9, color='#555555')
fig.tight_layout(rect=(0, .095, 1, .93), w_pad=3)
fig.savefig(ROOT / 'figures/cost_holdout_comparison.png', dpi=180)
plt.close(fig)
