import json
from pathlib import Path
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'q3_model_v2'))
from validate_model import independent_cells

data=json.loads((ROOT/'q3_model_v2/results/official_practice_20260911_analysis.json').read_text(encoding='utf-8'))
out=ROOT/'q3_model_v2/figures'
out.mkdir(exist_ok=True)
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','SimHei','DejaVu Sans'],
    'axes.unicode_minus':False, 'font.size':10, 'axes.spines.top':False,
    'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})

fig,axs=plt.subplots(1,2,figsize=(12,4.9),gridspec_kw={'width_ratios':[1.7,1]})
y=np.arange(5)
labels=[c['case'].split('-')[-1]+'号 · '+str(c['cleared'])+'个源' for c in data['cases']]
left=np.zeros(5)
for key,name,color in [('movement','移动','#607d94'),('measure','测向','#8aaba6'),
    ('switch','切频','#c5b17b'),('clear_fail','失败清除','#ca877b'),('clear_success','成功清除','#a6a0b1')]:
    widths=np.array([c['costs_s'][key] for c in data['cases']])
    axs[0].barh(y,widths,left=left,height=.6,label=name,color=color)
    left+=widths
for i,c in enumerate(data['cases']):
    axs[0].text(left[i]+28,i,f"{c['total_s']:.1f}",va='center',fontsize=10)
axs[0].set(yticks=y,yticklabels=labels,xlabel='虚拟任务时间 / 秒',xlim=(0,3570),title='五次官方演练：耗时组成')
axs[0].invert_yaxis()
fig.legend(*axs[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.085),ncol=5,frameon=False,fontsize=9)
axs[0].grid(axis='x',alpha=.17)
axs[0].set_axisbelow(True)
for i,c in enumerate(data['cases']):
    axs[1].barh(i,c['scan_tail_s'],height=.6,color='#c78d65')
    axs[1].text(c['scan_tail_s']+5,i,f"{c['scan_tail_s']:.1f}",va='center')
axs[1].set(yticks=y,yticklabels=labels,xlim=(0,340),xlabel='最后成功清除之后的时间 / 秒',title='三局仍有明显的查漏尾段')
axs[1].invert_yaxis()
axs[1].grid(axis='x',alpha=.17)
axs[1].set_axisbelow(True)
fig.text(.5,.015,'依据已接受的官方接口反馈独立核算；这些是不同局面，不能直接用于计算新旧版本的提速率。',ha='center',fontsize=9,color='#555555')
fig.tight_layout(rect=(0,.23,1,1),w_pad=2.8)
fig.savefig(out/'official_practice_costs.png',dpi=190)
plt.close(fig)

c=data['cases'][-1]
centers=independent_cells()
covered=np.zeros(len(centers),bool)
absent_channel=next(int(k) for k,v in c['independent_completion']['per_channel'].items() if not v['success_observed'])
for e in c['actions']:
    if e['time_s']<=c['last_clear_s']+1e-7 and e['channel']==absent_channel and e.get('measure_result')=='no_signal':
        covered |= np.linalg.norm(centers-e['position'],axis=1)+10*np.sqrt(2)<=1000
inside=(np.linalg.norm(centers,axis=1)<=1800)
fig,ax=plt.subplots(figsize=(8.5,7.7))
ax.add_patch(Circle((0,0),1800,fill=False,color='#7f8c94',lw=1.1))
q=centers[(~covered)&inside]
ax.scatter(q[:,0],q[:,1],s=4,marker='s',color='#eed7c5',alpha=.8,label='最后清除时仍待排除的区域')
points=np.vstack(([0,0],[e['position'] for e in c['actions']]))
ax.plot(points[:,0],points[:,1],color='#81949e',lw=1.1,alpha=.75,label='已执行路线',zorder=2)
successes=np.array([e['position'] for e in c['actions'] if e.get('clear_result')=='success'])
ax.scatter(successes[:,0],successes[:,1],s=34,c='#557e68',label='成功清除位置（非真源坐标）',zorder=4)
for i,b in enumerate(c['scan_batches'],1):
    x,y=b['position']
    ax.scatter([x],[y],marker='s',s=49,color='#496d91',zorder=5)
    dx,dy=(35,-85) if i==6 else (35,35)
    ax.text(x+dx,y+dy,f'S{i}',fontsize=10,color='#34526d',zorder=6)
last=next(e for e in reversed(c['actions']) if e.get('clear_result')=='success')
final=c['scan_batches'][-1]
start=np.array(last['position']); end=np.array(final['position'])
ax.annotate('',xy=end,xytext=start,arrowprops={'arrowstyle':'->','color':'#be7748','lw':2.5},zorder=5)
ax.scatter([start[0]],[start[1]],s=130,facecolors='none',edgecolors='#be7748',lw=1.8,zorder=6)
ax.annotate('最后清除',xy=start,xytext=(start[0]-750,start[1]+430),
    arrowprops={'arrowstyle':'-','color':'#be7748'},color='#9b5e37',fontsize=11)
ax.text(1520,750,'最后一段\n1224.8米移动\n+ 7次检测\n= 287.0秒',fontsize=11,color='#9b5e37',bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
ax.set(xlim=(-2000,2250),ylim=(-2000,2000),aspect='equal',xlabel='横坐标 / 米',ylabel='纵坐标 / 米',
    title='07号演练：完成清除后，仍需向东补齐覆盖')
ax.grid(alpha=.12)
ax.legend(loc='lower center',bbox_to_anchor=(.5,-.18),ncol=2,frameon=False,fontsize=9)
fig.text(.5,.017,'S1–S7为实际扫描站。淡色区域为20米方格证书的残余区，含保守近似；不表示那里实际存在源。',ha='center',fontsize=8.5,color='#555555')
fig.tight_layout(rect=(0,.075,1,1))
fig.savefig(out/'official_practice_07_route.png',dpi=180)
plt.close(fig)
print('Saved official_practice_costs.png and official_practice_07_route.png')
