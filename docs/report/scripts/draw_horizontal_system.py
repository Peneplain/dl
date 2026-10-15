"""Draw the shared text-to-motion pipeline as an editable vector figure."""
from pathlib import Path
import hashlib
import json
import os
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
TEXT = '#203549'
ARROW = '#425D70'
PALETTE = {
    'shared': ('#F1F5F9', '#667789'),
    'frozen': ('#E8F1FA', '#4F7BA2'),
    'physical': ('#E4F4E9', '#2E8B57'),
}

def draw():
    out = ROOT / 'figures'
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                         'pdf.fonttype':42, 'svg.fonttype':'none',
                         'mathtext.fontset':'dejavusans'})
    fig = plt.figure(figsize=(7, 2.62), facecolor='white')
    ax = fig.add_axes((0,0,1,1))
    ax.set(xlim=(0,14),ylim=(0,5.24))
    ax.axis('off')

    def arrow(points, color=ARROW, dashed=False, lw=1.15):
        for start,end in zip(points[:-2],points[1:-1]):
            ax.plot([start[0],end[0]],[start[1],end[1]],color=color,lw=lw,
                    linestyle=(0,(4,2)) if dashed else '-',solid_capstyle='round',zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
                     mutation_scale=10,color=color,lw=lw,
                     linestyle=(0,(4,2)) if dashed else '-',
                     shrinkA=0,shrinkB=1,zorder=2))

    def box(x,w,title,detail,color='shared',y=2.02,h=1.35):
        fill,edge=PALETTE[color]
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.10',
                                   fc=fill,ec=edge,lw=1.25,zorder=3))
        ax.text(x+w/2,y+h*.66,title,ha='center',va='center',fontsize=9.0,
                fontweight='bold',color=TEXT,linespacing=1.18,zorder=4)
        if detail:
            ax.text(x+w/2,y+h*.25,detail,ha='center',va='center',fontsize=7.7,
                    color=TEXT,linespacing=1.18,zorder=4)

    box(.12,1.78,'Language\n+ task','Sequencer')
    box(2.28,2.05,'ARDY /\nKimodo','Frozen','frozen')
    box(4.73,2.22,'G1 adapter','Reference buffer')
    box(7.35,1.75,'Reference','Validation')
    box(9.50,1.75,'SONIC','Frozen','frozen')
    box(11.65,2.20,'G1 / MuJoCo','Block + hand','physical')
    for left,right in [(1.90,2.28),(4.33,4.73),(6.95,7.35),(9.10,9.50),(11.25,11.65)]:
        arrow([(left,2.695),(right,2.695)])
    # A separate hand pathway preserves the shared finger controller.
    box(11.25,2.60,'Finger controller','Shared commands',y=4.00,h=.90)
    arrow([(1.01,3.37),(1.01,4.45),(11.25,4.45)])
    ax.text(6.10,4.53,'Finger timing / commands',ha='center',va='bottom',
            fontsize=8,color=ARROW,bbox={'fc':'white','ec':'none','pad':1.4})
    arrow([(12.55,4.00),(12.55,3.37)])
    ax.text(12.77,3.67,'hand',ha='left',va='center',fontsize=7.5,color=ARROW)
    # The feedback rail returns measured execution to task coordination.
    arrow([(12.75,2.02),(12.75,.70),(1.01,.70),(1.01,2.02)],dashed=True,lw=1.0)
    ax.text(7.00,.70,'Measured state + contact / alignment feedback',ha='center',va='center',
            fontsize=8.1,color=ARROW,bbox={'fc':'white','ec':'none','pad':2.0})
    # A small start cue identifies the entry without adding another module.
    ax.text(.12,3.67,'START',ha='left',va='center',fontweight='bold',fontsize=7.4,color='#667789')
    ax.text(5.84,1.71,r'Nominal reference $A$',ha='center',va='center',fontsize=7.9,color=ARROW)
    ax.text(10.37,1.71,'Whole-body control',ha='center',va='center',fontsize=7.9,color=ARROW)

    records=[]
    for stem in ('system','generator_pipeline'):
        for extension in ('pdf','png','svg'):
            target=out/f'{stem}.{extension}'
            fig.savefig(target,dpi=300,facecolor='white')
            records.append({'path':str(target.relative_to(ROOT)),
                            'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
    plt.close(fig)
    metadata={'schema':'report-horizontal-system-diagram-v1',
              'source':'scripts/draw_horizontal_system.py','size_inches':[7,2.62],
              'design':'Horizontal nominal route, separate shared finger control, and a measured-feedback rail. Frozen generators/controllers are blue, shared deterministic components gray, physical scene green.',
              'omitted_from_figure':'Network sizes, raw tensor dimensions, numeric control bounds and runtime frequencies remain in manuscript methods.',
              'new_physics':False,'new_model_inference':False,'files':records}
    (out/'horizontal-system-provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return records

if __name__=='__main__':
    draw()
    print('Created horizontal system diagrams (PDF, SVG, PNG).')
