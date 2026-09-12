"""Static local-experiment plots; field repetitions are averaged within layouts.

Example:
  python -X utf8 plot_cell_experiment.py --directory results/my-cells \
      --variant context --out-prefix figures/local-cell-comparison
No solver, simulator, or network is imported. Only PNG/SVG artifacts are written.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


POPULATION_NAMES = dict(uniform='均匀分布', boundary='边界分布', cluster='聚集分布',
                        omni_heavy='全向占多数', dir_heavy='定向占多数')
VARIANT_NAMES = dict(cells='22站覆盖', context='22站＋后续路线成本',
                     interleave='22站＋交替规划', opscan='22站＋交替规划及机会扫描',
                     resume25='25站＋逐次重排', share25='25站＋停点信息复用')
COSTS = [('move','移动','#566F89'), ('measure','检测','#74A79E'),
         ('switch','切频','#DBC48B'), ('clear_success','成功清除','#A59BB8'),
         ('clear_fail','失败清除','#C58C7E')]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def prepare(directory, variant):
    rows_path = directory/'rows.json'
    rows, analysis = read(rows_path), read(directory/'analysis.json')
    digest = analysis.get('provenance',{}).get('input_metadata_sha256',{}).get('rows.json')
    if digest and digest != hashlib.sha256(rows_path.read_bytes()).hexdigest():
        raise ValueError('rows.json changed after analysis.json; rerun the analyzer first')
    if variant == 'baseline25' or variant not in analysis['comparisons_to_baseline25']:
        raise ValueError('Select a candidate present in the paired analysis')
    if not all(analysis['safety_checks'].values()):
        raise ValueError('Plot requires a completed experiment whose safety checks passed')
    grouped = defaultdict(lambda:defaultdict(list))
    keys = defaultdict(set)
    for row in rows:
        if row['variant'] not in ('baseline25', variant):
            continue
        layout = row['seed'], row['population']
        grouped[layout][row['variant']].append(row)
        key = row['seed'], row['population'], row['field']
        if key in keys[row['variant']]:
            raise ValueError('Repeated variant/condition key')
        keys[row['variant']].add(key)
    if keys['baseline25'] != keys[variant]:
        raise ValueError('Baseline/candidate conditions are not exactly paired')
    records = []
    for (seed, population), by_variant in sorted(grouped.items()):
        if set(by_variant) != {'baseline25', variant}:
            raise ValueError('A layout is missing a paired variant')
        source_counts = {r['source_count'] for rr in by_variant.values() for r in rr}
        if len(source_counts) != 1:
            raise ValueError('Paired fields or variants have different source counts')
        item = dict(seed=seed, population=population, source_count=source_counts.pop())
        for version, rr in by_variant.items():
            item[version] = dict(
                mean_per_source_s=mean(r['total_s']/r['source_count'] for r in rr),
                mean_total_s=mean(r['total_s'] for r in rr),
                time_parts_s={k:mean(r['time_parts_s'].get(k,0.) for r in rr) for k,_,_ in COSTS},
                fields=sorted(r['field'] for r in rr))
        records.append(item)
    if not records:
        raise ValueError('No paired layouts')
    return records, len(keys[variant]), analysis


def plot(directory, variant, out_prefix):
    records, conditions, analysis = prepare(directory, variant)
    available = {f.name for f in font_manager.fontManager.ttflist}
    preferred = next((n for n in ('Microsoft YaHei','SimHei','Noto Sans CJK SC') if n in available),None)
    plt.rcParams.update({'font.family':preferred or 'DejaVu Sans', 'axes.unicode_minus':False,
        'font.size':10, 'axes.titlesize':12, 'axes.labelsize':10,
        'svg.fonttype':'path', 'savefig.facecolor':'#FAFAF7'})
    baseline_color, candidate_color = '#607B96', '#AD704B'
    candidate_name = VARIANT_NAMES.get(variant, variant)
    fig, (ax, cost_ax) = plt.subplots(1,2,figsize=(14.5,6.5),gridspec_kw={'width_ratios':[1.5,1]})
    fig.patch.set_facecolor('#FAFAF7')
    for axes in (ax, cost_ax):
        axes.set_facecolor('#FAFAF7')
        axes.spines[['top','right']].set_visible(False)
        axes.spines[['left','bottom']].set_color('#B5B7B3')
        axes.tick_params(color='#B5B7B3')
        axes.grid(axis='y',alpha=.35,color='#D8DBD6',linewidth=.7,zorder=0)
        axes.set_axisbelow(True)
    x = np.arange(len(records))
    old = np.array([r['baseline25']['mean_per_source_s'] for r in records])
    new = np.array([r[variant]['mean_per_source_s'] for r in records])
    ax.vlines(x,np.minimum(old,new),np.maximum(old,new),color='#B6BBB6',linewidth=2,zorder=2)
    ax.scatter(x-.025,old,s=57,c=baseline_color,marker='o',edgecolors='#FAFAF7',linewidth=.8,zorder=3)
    ax.scatter(x+.025,new,s=62,c=candidate_color,marker='D',edgecolors='#FAFAF7',linewidth=.8,zorder=4)
    ax.set_xticks(x,[f'{r["seed"]}\n{POPULATION_NAMES.get(r["population"],r["population"])}' for r in records],fontsize=8.5)
    ax.set_xlim(-.6,len(records)-.4)
    ax.set_ylim(0,max(float(old.max()),float(new.max()))*1.16)
    ax.set_ylabel('单源虚拟时间（秒/源）')
    ax.set_title(f'每个独立布局的配对结果（{len(records)}个布局）',loc='left',pad=17,fontweight='bold')
    ax.legend(handles=[Line2D([],[],color=baseline_color,marker='o',linestyle='',label='25站基线'),
                       Line2D([],[],color=candidate_color,marker='D',linestyle='',label=candidate_name)],
              frameon=False,loc='upper left',fontsize=9)
    paired_savings = old-new
    wins,ties,losses = (int(np.sum(paired_savings>1e-6)),int(np.sum(np.abs(paired_savings)<=1e-6)),int(np.sum(paired_savings < -1e-6)))
    ax.text(.99,.975,f'按布局：{wins}胜 / {ties}平 / {losses}负',ha='right',va='top',transform=ax.transAxes,fontsize=9,color='#59615B')
    versions = ('baseline25',variant)
    bottom = np.zeros(2)
    for component,label,color in COSTS:
        values = np.array([mean(r[v]['time_parts_s'][component] for r in records) for v in versions])
        cost_ax.bar(np.arange(2),values,bottom=bottom,color=color,width=.53,label=label,zorder=3)
        for i,value in enumerate(values):
            if value > 350:
                cost_ax.text(i,bottom[i]+value/2,f'{value:,.0f}',ha='center',va='center',color='white' if component=='move' else '#263B36',fontsize=10)
        bottom += values
    for i,total in enumerate(bottom):
        cost_ax.text(i,total+max(bottom)*.022,f'{total:,.1f}秒',ha='center',va='bottom',fontweight='bold',fontsize=11)
    cost_ax.set_ylim(0,max(bottom)*1.17)
    cost_ax.set_xlim(-.65,1.65)
    cost_ax.set_xticks([0,1],['25站基线',candidate_name],fontsize=9)
    cost_ax.set_ylabel('每局总虚拟时间的布局均值（秒）')
    cost_ax.set_title('完整任务的费用分解',loc='left',pad=17,fontweight='bold')
    cost_ax.legend(frameon=False,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.125),fontsize=9,columnspacing=1.1)
    mean_old,mean_new = (mean(r[v]['mean_total_s'] for r in records) for v in versions)
    percent = 100*(mean_old-mean_new)/mean_old
    fig.suptitle('第四问：本地配对模拟结果',x=.07,y=.98,ha='left',fontsize=18,fontweight='bold',color='#273C37')
    fig.text(.07,.921,f'{len(records)}个独立布局，{conditions}组配对条件  |  候选：{candidate_name}  |  布局平均总时间变化：{-percent:+.2f}%',fontsize=11,color='#58635D')
    fig.text(.07,.045,'同一布局先对各误差场取均值，再进行布局间汇总；不同误差场不计为独立布局。',fontsize=9,color='#67706A')
    fig.text(.07,.018,'仅本地模拟，非官方测试。单源时间按该布局的全部源数计算；费用分解包含移动、检测、切频和所有清除操作。',fontsize=9,color='#67706A')
    fig.subplots_adjust(left=.07,right=.975,top=.815,bottom=.215,wspace=.3)
    out_prefix.parent.mkdir(parents=True,exist_ok=True)
    outputs=[]
    for suffix in ('.png','.svg'):
        path=Path(str(out_prefix)+suffix)
        fig.savefig(path,dpi=200)
        outputs.append(path)
    plt.close(fig)
    print(f'{len(records)} independent layouts / {conditions} paired conditions; font={preferred or "DejaVu Sans"}')
    print(f'Layout means: baseline={mean_old:.3f}s, candidate={mean_new:.3f}s, reduction={percent:.3f}%')
    for path in outputs:
        print(path)
    return outputs


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',required=True,type=Path)
    parser.add_argument('--variant',required=True)
    parser.add_argument('--out-prefix',required=True,type=Path)
    args=parser.parse_args()
    plot(args.directory,args.variant,args.out_prefix)
