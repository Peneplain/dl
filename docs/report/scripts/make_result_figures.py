"""Rebuild published result charts from results/public-results.json.

Only recorded descriptive values and stored statistical results are displayed.
This script does not run experiments, train models, or repeat inferential tests.
"""
from pathlib import Path
import argparse
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / 'figures'
DATA = None
RESULTS_PATH = ROOT / 'results/public-results.json'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 8.4,
    'axes.titlesize': 9.2, 'axes.labelsize': 8.4,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#667585', 'axes.linewidth': .65,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.facecolor': 'white', 'text.color': '#203244',
    'axes.labelcolor': '#203244', 'xtick.color': '#203244',
    'ytick.color': '#203244', 'mathtext.fontset': 'dejavusans',
})
INK = '#203244'
BLUE = '#32658D'
TEAL = '#138276'
ORANGE = '#C27229'
GREY = '#657383'
PALE = '#EFF4F8'
GREEN = '#EAF5F1'
AMBER = '#FFF4E8'
LINE = '#CED9E3'
PURPLE = '#765F9E'

def save(fig, name, dpi=260):
    # Fixed seven-inch figures preserve the authored type sizes in two columns.
    fig.savefig(FIG / f'{name}.pdf')
    fig.savefig(FIG / f'{name}.png', dpi=dpi)
    plt.close(fig)

def label(ax, x, y, text, size=7.5, color=GREY, **kwargs):
    ax.text(x, y, text, ha=kwargs.pop('ha', 'center'),
            va=kwargs.pop('va', 'center'), color=color, fontsize=size,
            zorder=5, **kwargs)

def confirmation():
    study = DATA['controller_confirmation']
    tally = [[0, 0], [0, 0]]
    for row in study['trials']:
        if row['complete_end_to_end_pair']:
            a = row['end_to_end_success']
            tally[int(a['ardy'])][int(a['ours'])] += 1
    n = sum(map(sum, tally)); baseline = sum(tally[1]); ours = tally[0][1] + tally[1][1]
    assert n == study['result']['complete_pairs']
    assert baseline == study['result']['B0_successes']
    assert ours == study['result']['candidate_successes']
    fig, axs = plt.subplots(1, 2, figsize=(7, 2.27), gridspec_kw={'width_ratios': [1.1, 1]})
    ax = axs[0]
    vals = [100*baseline/n, 100*ours/n]
    ax.barh([1, 0], vals, color=[BLUE, TEAL], height=.49)
    ax.set_yticks([1, 0], ['ARDY baseline', 'Ours'])
    ax.set_xlim(0, 81); ax.set_ylim(-.60, 1.65)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel('End-to-end grasp success (%)')
    for y, v, k in zip([1, 0], vals, [baseline, ours]):
        ax.text(v+1.4, y, f'{k}/{n}', va='center', fontsize=8.8)
    ax.set_title('(a) Independent confirmation', loc='left', fontweight='bold', pad=10)
    ax.grid(axis='x', alpha=.17); ax.set_axisbelow(True)
    ax = axs[1]; ax.axis('off'); ax.set_xlim(0, 5.2); ax.set_ylim(0, 3.2)
    ax.set_title('(b) Same-origin outcomes', loc='left', fontweight='bold', pad=10)
    label(ax, 2.91, 2.85, 'Ours\nfails', 8.1, INK)
    label(ax, 4.32, 2.85, 'Ours\nsucceeds', 8.1, INK)
    label(ax, 1.70, 2.07, 'Baseline fails', 8.1, INK, ha='right')
    label(ax, 1.70, 1.03, 'Baseline succeeds', 8.1, INK, ha='right')
    x0, w, h = 2.13, 1.40, 1.00
    cells = [(x0, 1.57, str(tally[0][0]), PALE), (x0+w, 1.57, f'{tally[0][1]}\nrescues', GREEN),
             (x0, .53, f'{tally[1][0]}\nharms', AMBER), (x0+w, .53, str(tally[1][1]), PALE)]
    for x, y, t, fc in cells:
        ax.add_patch(Rectangle((x,y), w,h,fc=fc,ec=LINE,lw=.7))
        label(ax,x+w/2,y+h/2,t,9.4,INK,linespacing=1.35)
    label(ax, 3.53, .13, f'Observed gain: {100*(ours-baseline)/n:+.1f} pp', 7.7, TEAL)
    fig.subplots_adjust(left=.135, right=.985, bottom=.23, top=.79, wspace=.48)
    save(fig, 'confirmation')

def gating():
    study = DATA['invocation_study']
    arms = study['efficiency']['by_arm']
    selective = arms['selective']; always = arms['always_on']
    n = study['scheduled_origins']
    baseline_count = study['end_to_end']['selective_vs_ardy']['baseline_successes']
    counts = [baseline_count, selective['successful_origins'], always['successful_origins']]
    fig, axs = plt.subplots(1, 3, figsize=(7, 2.26), gridspec_kw={'width_ratios': [1.12,1,1]})
    ax=axs[0]
    rates=[100*k/n for k in counts]
    ax.bar([0,1,2],rates,color=[BLUE,TEAL,ORANGE],width=.60)
    ax.set_xticks([0,1,2],['ARDY\nbaseline','Selective','Always-on'])
    ax.set_ylim(0,87);ax.set_yticks([0,25,50,75]);ax.set_ylabel('End-to-end success (%)')
    for x,y,k in zip([0,1,2],rates,counts):
        ax.text(x,y+3.2,f'{k}/{n}',ha='center',fontsize=8.4)
    ax.set_title('(a) Grasp success',loc='left',fontweight='bold',pad=9)
    ax=axs[1]
    inv=[selective['residual_activation_fraction']*100,always['residual_activation_fraction']*100]
    ax.bar([0,1],inv,color=[TEAL,ORANGE],width=.57)
    ax.set_xticks([0,1],['Selective','Always-on']);ax.set_ylim(0,121)
    ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Residual invocation (%)')
    for x,y in enumerate(inv):ax.text(x,y+4,f'{y:.2f}%' if y<100 else '100%',ha='center',fontsize=8.4)
    ax.set_title('(b) Computation',loc='left',fontweight='bold',pad=9)
    ax=axs[2]
    times=[selective['timings_ms']['controller_ms']['mean'],always['timings_ms']['controller_ms']['mean']]
    ax.bar([0,1],times,color=[TEAL,ORANGE],width=.57)
    ax.set_xticks([0,1],['Selective','Always-on']);ax.set_ylim(0,6.6)
    ax.set_yticks([0,2,4,6]);ax.set_ylabel('Mean controller time (ms)')
    for x,y in enumerate(times):ax.text(x,y+.20,f'{y:.2f}',ha='center',fontsize=8.4)
    ax.set_title('(c) Controller latency',loc='left',fontweight='bold',pad=9)
    axs[0].tick_params(axis='x',labelsize=7.2)
    for ax in axs:ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fig.subplots_adjust(left=.073,right=.985,bottom=.24,top=.80,wspace=.60)
    save(fig,'gating')

def generator_comparison(include_failure_profile=True):
    data=DATA['generator_comparison']['methods']; keys=['ardy','kimodo']
    funnel={k:data[k]['retention'] for k in keys}
    # Validate published descriptive accounting without requiring private files.
    for k in keys:
        f=funnel[k]; rows=data[k]['trials']
        assert f['scheduled']==len(rows)==data[k]['evaluated']==100
        assert sum(r['initial_threshold_reached'] for r in rows)==f['initial_threshold']
        assert sum(r['sticky_loss_after_threshold'] for r in rows)==f['sticky_loss']
        assert sum(r['success'] for r in rows)==f['final_successes']==data[k]['successes']
        assert f['initial_threshold']-f['sticky_loss']==f['final_successes']
    fig,axs=plt.subplots(1,3,figsize=(7,2.42))
    colors=[BLUE,PURPLE]
    ax=axs[0]
    rates=[100*data[k]['success_rate'] for k in keys]
    lows=[rates[i]-100*data[k]['success_wilson_95'][0] for i,k in enumerate(keys)]
    highs=[100*data[k]['success_wilson_95'][1]-rates[i] for i,k in enumerate(keys)]
    ax.bar([0,1],rates,color=colors,width=.56,
           yerr=[lows,highs],capsize=3,error_kw={'lw':1,'capthick':1,'ecolor':INK})
    ax.set_xticks([0,1],['ARDY','Kimodo']);ax.set_ylim(0,82);ax.set_yticks([0,20,40,60,80])
    ax.set_ylabel('Final success (%)')
    for x,y,k in zip([0,1],rates,keys):
        ax.text(x,100*data[k]['success_wilson_95'][1]+3.8,
                f'{data[k]["successes"]}/100',ha='center',fontsize=8.7,fontweight='bold')
    ax.set_title('(a) Final task success',loc='left',fontweight='bold',pad=21)
    label(ax,0,1.045,'95% Wilson intervals',7.2,GREY,
          ha='left',va='bottom',transform=ax.transAxes)
    ax=axs[1]
    retained=[100*funnel[k]['conditional_retention_rate'] for k in keys]
    lost=[100*funnel[k]['conditional_loss_rate'] for k in keys]
    ax.bar([0,1],retained,color=colors,width=.56)
    ax.bar([0,1],lost,bottom=retained,color=ORANGE,width=.56)
    for x,k,n,d in zip([0,1],keys,retained,lost):
        f=funnel[k]
        ax.text(x,n/2,f'{n:.1f}%\n{f["final_successes"]}/{f["initial_threshold"]}',
                ha='center',va='center',color='white',fontsize=8.0,fontweight='bold',linespacing=1.3)
        if d>15:
            ax.text(x,n+d/2,f'{f["sticky_loss"]} lost',ha='center',va='center',
                    color='white',fontsize=7.5,fontweight='bold')
        else:
            ax.annotate(f'{f["sticky_loss"]} lost',xy=(x,100),xytext=(x,108),
                    ha='center',va='center',fontsize=7.5,color=ORANGE,fontweight='bold',
                    arrowprops={'arrowstyle':'-','color':ORANGE,'lw':.7,'shrinkA':2,'shrinkB':2})
    ax.set_xticks([0,1],['ARDY','Kimodo']);ax.set_ylim(0,116);ax.set_yticks([0,25,50,75,100])
    ax.set_ylabel('Retention after threshold (%)')
    ax.set_title('(b) After the initial lift',loc='left',fontweight='bold',pad=21)
    label(ax,0,1.045,'58 initially qualify / method',7.2,GREY,
          ha='left',va='bottom',transform=ax.transAxes)
    ax=axs[2]
    values=[data[k]['metrics']['command_to_first_reference_wall_seconds']['median'] for k in keys]
    ax.bar([0,1],values,color=colors,width=.56)
    ax.set_xticks([0,1],['ARDY','Kimodo']);ax.set_ylim(0,10.6);ax.set_yticks([0,2,4,6,8,10])
    ax.set_ylabel('First-reference response (s)')
    for x,y in enumerate(values):ax.text(x,y+.38,f'{y:.2f} s',ha='center',fontsize=8.7,fontweight='bold')
    ax.set_title('(c) Median response',loc='left',fontweight='bold',pad=21)
    label(ax,0,1.045,'Command → first reference',7.2,GREY,
          ha='left',va='bottom',transform=ax.transAxes)
    for ax in axs:ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    fig.subplots_adjust(left=.072,right=.985,bottom=.22,top=.77,wspace=.65)
    save(fig,'generator-comparison')
    if not include_failure_profile:return
    # Failure categories retain all 100 scheduled trials per method as denominator.
    reason_groups=[('grasp_alignment_missed','Grasp alignment missed'),('timeout','Timeout'),('grasp_not_acquired','Grasp not acquired'),
                   ('grasp_lost_after_success','Grasp lost after success'),
                   ('prohibited_robot_table_contact','Robot–table contact'),
                   ('prohibited_robot_block_contact','Robot–block contact'),
                   ('block_floor_contact','Block–floor contact'),('lower_error','Lower-stage error'),('lift_error','Lift-stage error')]
    fig,ax=plt.subplots(figsize=(7,3.12))
    y=list(range(len(reason_groups)))
    ax.barh([v+.18 for v in y],[data['ardy']['failure_reasons'].get(k,0) for k,_ in reason_groups],
            height=.34,color=BLUE,label='ARDY')
    ax.barh([v-.18 for v in y],[data['kimodo']['failure_reasons'].get(k,0) for k,_ in reason_groups],
            height=.34,color=PURPLE,label='Kimodo')
    assert set().union(*(set(data[k]['failure_reasons']) for k in keys)) == {k for k,_ in reason_groups}
    ax.set_yticks(y,[v for _,v in reason_groups]);ax.invert_yaxis()
    ax.set_xlabel('Failed trials out of 100 scheduled per method');ax.set_xlim(0,31)
    ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True);ax.legend(frameon=False,loc='lower right',ncol=2)
    fig.subplots_adjust(left=.28,right=.985,bottom=.18,top=.96)
    save(fig,'failure-profile')

def learning_curves():
    fig,axs=plt.subplots(1,2,figsize=(7,2.45))
    for ax,kind,tag in zip(axs,['risk','residual'],['(a)','(b)']):
        report=DATA['selected_training'][kind]
        rows=report['full_epoch_history']; epochs=[r['epoch'] for r in rows]
        best=report['best_epoch']; assert len(rows)==report['actual_epochs']
        val=next(r['val_loss'] for r in rows if r['epoch']==best)
        assert abs(val-report['best_val_loss'])<1e-12,(kind,val)
        ax.plot(epochs,[r['train_loss'] for r in rows],color=BLUE,lw=1.4,label='Training')
        ax.plot(epochs,[r['val_loss'] for r in rows],color=TEAL,lw=1.5,label='Validation')
        ax.axvline(best,color=GREY,lw=.85,ls='--',alpha=.8)
        ax.scatter([best],[val],color=TEAL,edgecolor='white',s=32,lw=.7,zorder=5)
        ax.annotate(f'Retained epoch {best}',xy=(best,val),
                    xytext=(.20,.65) if kind=='risk' else (.44,.65),textcoords='axes fraction',
                    fontsize=8,color=INK,bbox={'facecolor':'white','edgecolor':'none','alpha':.86,'pad':1.5},
                    arrowprops={'arrowstyle':'-','lw':.7,'color':GREY})
        ax.set_yscale('log');ax.set_xlim(.5,len(rows)+.5)
        ax.set_xlabel('Training epoch');ax.set_ylabel('Recorded supervised objective')
        ax.set_title(f'{tag} {kind.title()} · {len(rows)} epochs',loc='left',fontweight='bold',pad=10)
        ax.grid(axis='both',which='major',alpha=.16);ax.set_axisbelow(True)
        ax.legend(frameon=False,fontsize=7.8,loc='upper right')
        if kind=='risk':ax.set_xticks([1,5,10,15,19]);ax.set_ylim(.14,1.25)
        else:ax.set_xticks([1,10,20,30,40,45]);ax.set_ylim(1.7e-4,2.1e-2)
        ax.yaxis.set_minor_formatter(NullFormatter())
    fig.subplots_adjust(left=.09,right=.985,bottom=.20,top=.81,wspace=.36)
    save(fig,'learning-curves')

def failure_profile_column():
    data=DATA['generator_comparison']['methods']
    pairs=[('grasp_alignment_missed','Alignment miss'),('timeout','Timeout'),
           ('grasp_not_acquired','No grasp acquired'),('grasp_lost_after_success','Grasp lost'),
           ('prohibited_robot_table_contact','Table contact'),
           ('prohibited_robot_block_contact','Block contact'),('block_floor_contact','Floor contact'),
           ('lower_error','Lower-stage error'),('lift_error','Lift-stage error')]
    assert set().union(*(set(v['failure_reasons']) for v in data.values()))=={k for k,_ in pairs}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7.8,'axes.labelsize':7.8,
                         'xtick.labelsize':7.5,'ytick.labelsize':7.5,'pdf.fonttype':42,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'axes.edgecolor':GREY,'axes.linewidth':.65})
    fig,ax=plt.subplots(figsize=(3.4,3.12));ys=list(range(len(pairs)))
    for method,delta,color,label in [('ardy',.18,BLUE,'ARDY'),('kimodo',-.18,PURPLE,'Kimodo')]:
        ax.barh([y+delta for y in ys],[data[method]['failure_reasons'].get(k,0) for k,_ in pairs],
                height=.33,color=color,label=label)
    ax.set_yticks(ys,[v for _,v in pairs]);ax.invert_yaxis();ax.set_xlim(0,31)
    ax.set_xticks([0,10,20,30]);ax.set_xlabel('Failed trials (out of 100 per method)')
    ax.grid(axis='x',alpha=.16);ax.set_axisbelow(True);ax.legend(frameon=False,loc='lower right',fontsize=7.5)
    fig.subplots_adjust(left=.345,right=.985,top=.97,bottom=.16)
    fig.savefig(FIG/'failure-profile-column.pdf')
    fig.savefig(FIG/'failure-profile-column.png',dpi=300);plt.close(fig)


def main():
    global DATA, FIG, RESULTS_PATH
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=RESULTS_PATH)
    parser.add_argument('--output',type=Path,default=FIG)
    args=parser.parse_args()
    RESULTS_PATH=args.results.resolve(); FIG=args.output.resolve()
    DATA=json.loads(RESULTS_PATH.read_text()); FIG.mkdir(parents=True,exist_ok=True)
    assert DATA['schema']=='dl-public-report-results-v1'
    generator_comparison()
    confirmation()
    gating()
    learning_curves()
    failure_profile_column()
    names=['generator-comparison','failure-profile','confirmation','gating',
           'learning-curves','failure-profile-column']
    files=[FIG/f'{name}.{ext}' for name in names for ext in ['pdf','png']]
    receipt={
        'schema':'public-report-result-figure-regeneration-v1',
        'source_results_sha256':hashlib.sha256(RESULTS_PATH.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'new_experiments':False,'new_training':False,'new_inferential_tests':False,
        'outputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    (FIG/'result-figures-regeneration.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Rebuilt six result figures (PDF and PNG) from public-results.json.')


if __name__=='__main__':
    main()
