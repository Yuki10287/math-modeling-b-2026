"""Standalone figure for the frozen global-value experiment."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT/'results/value_holdout_summary.json').read_text())
selected = data['selected']
comparison = data['comparisons_to_original'][selected]
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'], 'axes.unicode_minus':False,
    'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
labels = {'uniform':'普通分布', 'sparse':'10源稀疏', 'boundary':'边界分布', 'one-side':'单侧分布'}
colors = {'uniform':'#607d8b', 'sparse':'#7c9279', 'boundary':'#b99370', 'one-side':'#968aa3'}
fig, axes = plt.subplots(1,2,figsize=(12.5,5),gridspec_kw={'width_ratios':[1.4,1]})
for kind,label in labels.items():
    pairs = [p for p in data['pairs'] if p['population']==kind]
    axes[0].scatter([p['original_s'] for p in pairs], [p['saved_s'] for p in pairs],
        label=label+'（15组）',color=colors[kind],s=35,alpha=.8,edgecolors='white',linewidths=.4)
axes[0].axhline(0,color='#555555',linewidth=.8)
axes[0].set(xlabel='上一版联合调度总时间 / 秒',ylabel='节省时间 / 秒（正值表示候选版更快）',
    title=f"60组配对：{comparison['faster']}组更快，{comparison['slower']}组更慢")
axes[0].legend(frameon=False,fontsize=8,loc='lower right')
axes[0].grid(axis='y',alpha=.15)
means = [data['populations'][k]['comparisons_to_original'][selected]['mean_saved_s'] for k in labels]
axes[1].barh(list(labels.values())[::-1],means[::-1],color=list(colors.values())[::-1],height=.55)
axes[1].axvline(0,color='#555555',linewidth=.8)
axes[1].set(xlabel='平均节省时间 / 秒',title='每类5种新布局，各测试3种误差场',xlim=(min(-6,min(means)-5),max(means)+7))
for i,v in enumerate(means[::-1]):
    axes[1].text(v+(.4 if v>=0 else -.4),i,f'{v:+.2f}',va='center',ha='left' if v>=0 else 'right')
old,new = data['groups']['original']['mean_total_s'],data['groups'][selected]['mean_total_s']
fig.suptitle(f'第三问全局价值候选：{old:.2f} → {new:.2f}秒，平均节省{comparison["reduction_pct"]:.2f}%',fontsize=14,y=.98)
lo,hi = data['paired_saved_s_ci95']
fig.text(.02,.045,f'20种新布局，60组配对条件；按布局分层重抽样的平均节省95%区间：{lo:.2f}～{hi:.2f}秒。',fontsize=9,color='#555555')
fig.text(.02,.01,'本地自建环境；参照为上一版联合调度候选。单局可能退步；没有运行官方模拟器。',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.095,1,.93),w_pad=3)
fig.savefig(ROOT/'figures/value_holdout_comparison.png',dpi=180)
plt.close(fig)
