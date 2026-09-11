"""Standalone scientific figure for the frozen local paired experiment."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
data=json.loads((ROOT/'results'/'joint_holdout_summary.json').read_text(encoding='utf-8'))
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'],'axes.unicode_minus':False,
                     'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
labels={'uniform':'普通分布（36组）','sparse':'10源稀疏（8组）','boundary':'边界分布（8组）','one-side':'单侧分布（8组）'}
colors={'uniform':'#607d8b','sparse':'#7c9279','boundary':'#b99370','one-side':'#968aa3'}
fig,axes=plt.subplots(1,2,figsize=(12.5,4.7),gridspec_kw={'width_ratios':[1.45,1]})
for name,label in labels.items():
    pairs=[p for p in data['pairs'] if p['population']==name]
    axes[0].scatter([p['baseline_s'] for p in pairs],[p['saved_s'] for p in pairs],
                    label=label,color=colors[name],s=32,alpha=.8,edgecolors='white',linewidths=.4)
axes[0].axhline(0,color='#555555',linewidth=.8)
axes[0].set(xlabel='原 lean 版本总时间 / 秒',ylabel='节省时间 / 秒（正值表示新版更快）',title='60组配对结果：40组更快，20组更慢')
axes[0].legend(frameon=False,fontsize=8,loc='upper right')
axes[0].grid(axis='y',alpha=.15)
means=[data['populations'][n]['comparisons_to_lean']['route_interleave']['mean_saved_s'] for n in labels]
axes[1].barh(list(labels.values())[::-1],means[::-1],color=list(colors.values())[::-1],height=.57)
axes[1].axvline(0,color='#555555',linewidth=.8)
axes[1].set(xlabel='平均节省时间 / 秒',title='分组均值：边界分布略有退步',xlim=(-20,65))
for i,v in enumerate(means[::-1]):
    axes[1].text(v+(1 if v>=0 else -1),i,f'{v:+.1f}',va='center',ha='left' if v>=0 else 'right')
fig.suptitle('第三问联合调度候选版：平均3070.2 → 3036.7秒，降低1.09%',fontsize=14,y=.98)
fig.text(.02,.015,'本地自建环境；24种新布局，误差场按布局成组。没有运行官方模拟器；结果不保证每局或官方测试都更快。',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.065,1,.93),w_pad=3)
fig.savefig(ROOT/'figures'/'joint_holdout_comparison.png',dpi=180)
plt.close(fig)
