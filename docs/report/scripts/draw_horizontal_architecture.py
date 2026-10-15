"""Draw the deployed Risk-to-Residual control loop as a paper figure.

This script draws an architecture diagram only. It performs no policy inference,
training, physics steps, or statistical analysis. Its topology follows the
repository README and the preserved source used by the reported controller.
"""
from pathlib import Path
import hashlib
import json
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.path import Path as DrawPath

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / 'figures'
SIZE = (10.5, 3.5)
COLORS = {
    'frozen': ('#426A94', '#EAF1F8'),
    'risk': ('#087F78', '#E7F5F1'),
    'residual': ('#7756A4', '#F0EBF7'),
    'shared': ('#607487', '#F1F4F7'),
    'physics': ('#35835D', '#EAF5EE'),
    'limiter': ('#B47A23', '#FFF4DD'),
}
INK = '#23374A'
ARROW = '#405B70'
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11.5,
    'text.color': INK,
    'mathtext.fontset': 'dejavusans',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'svg.fonttype': 'none',
    'savefig.facecolor': 'white',
})


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label(ax, x, y, value, *, size=11.5, color=INK, bold=False, white=False,
          ha='center', va='center'):
    options = {}
    if white:
        options['bbox'] = {'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.5}
    ax.text(x, y, value, ha=ha, va=va, fontsize=size, color=color,
            fontweight='bold' if bold else 'normal', zorder=7, **options)


def box(ax, x, y, w, h, title, body, role, *, title_size=13.0, body_size=11.1):
    stroke, fill = COLORS[role]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle='round,pad=0.008,rounding_size=0.052',
        linewidth=1.1, edgecolor=stroke, facecolor=fill, zorder=4,
    ))
    label(ax, x+w/2, y+h*.72, title, size=title_size, color=stroke, bold=True)
    label(ax, x+w/2, y+h*.29, body, size=body_size)


def line(ax, points, *, color=ARROW, width=1.2, dashed=False, arrow=True, head=10):
    path = DrawPath(points, [DrawPath.MOVETO] + [DrawPath.LINETO]*(len(points)-1))
    ax.add_patch(FancyArrowPatch(
        path=path, arrowstyle='-|>' if arrow else '-', mutation_scale=head,
        linewidth=width, color=color, linestyle=(0, (4, 2.6)) if dashed else '-',
        shrinkA=0, shrinkB=0, joinstyle='round', capstyle='round', zorder=2,
    ))


def junction(ax, x, y, color=ARROW):
    ax.add_patch(Circle((x, y), .024, facecolor=color, edgecolor='white',
                        linewidth=.35, zorder=6))


def draw():
    FIGURES.mkdir(exist_ok=True)
    fig = plt.figure(figsize=SIZE, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, SIZE[0])
    ax.set_ylim(0, SIZE[1])
    ax.axis('off')
    frozen = COLORS['frozen'][0]
    teal = COLORS['risk'][0]
    purple = COLORS['residual'][0]
    gray = COLORS['shared'][0]
    amber = COLORS['limiter'][0]
    green = COLORS['physics'][0]

    # The top lane is the common nominal execution path. Nominal references
    # exist before SONIC and are not SONIC's motor-command output.
    box(ax, .20, 2.75, 1.20, .50, 'Language', 'Task grounding', 'shared',
        title_size=12.7, body_size=10.5)
    box(ax, 1.70, 2.75, 1.30, .50, 'Frozen ARDY', 'Motion generator', 'frozen',
        title_size=12.4, body_size=10.6)
    box(ax, 3.30, 2.75, 1.50, .50, 'Reference', 'Adapter + buffer', 'frozen',
        title_size=13.0, body_size=10.7)
    box(ax, 8.43, 2.75, .90, .50, 'SONIC', 'Frozen', 'frozen',
        title_size=13.0, body_size=10.7)
    box(ax, 9.65, 2.75, .65, .50, 'G1', 'MuJoCo', 'physics',
        title_size=13.0, body_size=10.4)
    line(ax, [(1.40, 3.00), (1.70, 3.00)])
    line(ax, [(3.00, 3.00), (3.30, 3.00)])
    line(ax, [(4.80, 3.00), (7.91, 3.00)], width=1.45)
    label(ax, 6.12, 3.14, r'Nominal $A$', size=12.0, color=frozen, white=True)
    line(ax, [(8.17, 3.00), (8.43, 3.00)], width=1.45)
    line(ax, [(9.33, 3.00), (9.65, 3.00)], width=1.45)

    # Summation is downstream of the offset filter, whose joint-relative bounds
    # also read the nominal reference. No correction changes the generator.
    ax.add_patch(Circle((8.04, 3.00), .125, facecolor='white', edgecolor=amber,
                        linewidth=1.35, zorder=5))
    label(ax, 8.04, 3.005, '+', size=19.0, color=amber)
    label(ax, 8.04, 3.22, r'$A+\delta$', size=12.3, color=amber, white=True)

    # Fingers bypass SONIC's body-joint reference path and share the task.
    line(ax, [(.80, 3.25), (.80, 3.38), (9.975, 3.38), (9.975, 3.25)],
         color=amber, width=1.05, head=8.5)
    label(ax, 6.20, 3.38, 'Separate articulated-finger control',
          size=11.4, color=amber, white=True)

    # The learned information pathway is the dominant central row.
    box(ax, .20, 1.10, 1.28, .74, r'History $S$', 'Measured\nexecution', 'shared',
        title_size=13.1, body_size=10.5)
    box(ax, 1.80, 1.10, 1.46, .74, 'Predictive Risk', 'Transformer', 'risk',
        title_size=12.2, body_size=11.5)
    box(ax, 3.53, 1.10, 1.62, .74, r'Structured $Z$', '', 'risk',
        title_size=13.2)
    label(ax, 4.34, 1.48, 'Body × future time', size=10.6, color=teal)
    # The four-by-eight miniature map depicts the retained indexing, not
    # measured heat values or a rendering of learned activation magnitudes.
    for body in range(4):
        for future in range(8):
            ax.add_patch(Rectangle((3.73+future*.153, 1.175+body*.052), .135, .040,
                                   facecolor='#C1E4DA', edgecolor=teal,
                                   linewidth=.35, zorder=6))
    box(ax, 5.42, 1.10, 1.35, .74, 'Residual', 'Transformer\n+ projected $Z$',
        'residual', title_size=13.8, body_size=10.7)
    box(ax, 7.13, 1.10, 1.53, .74, 'Limiter + checks',
        'Magnitude + slew\nArm + joint checks', 'limiter',
        title_size=11.8, body_size=10.4)
    line(ax, [(1.48, 1.47), (1.80, 1.47)], color=gray, width=1.25, head=9)
    line(ax, [(3.26, 1.47), (3.53, 1.47)], color=teal, width=2.05, head=11)
    line(ax, [(5.15, 1.47), (5.42, 1.47)], color=teal, width=2.65, head=12)
    line(ax, [(6.77, 1.47), (7.13, 1.47)], color=purple, width=1.7, head=10.5)
    label(ax, 6.95, 1.00, r'$\Delta A^{\rm req}$', size=11.4, color=purple,
          white=True)

    # Uncorrected future A branches to both independently parameterized models.
    # C is phase/finger context from the task coordinator and is specified in
    # the manuscript; the figure follows README's concise A/S topology.
    line(ax, [(4.99, 3.00), (4.99, 2.62)], color=frozen, width=1.0, arrow=False)
    line(ax, [(2.06, 2.62), (6.46, 2.62)], color=frozen, width=1.0, arrow=False)
    line(ax, [(2.06, 2.62), (2.06, 1.84)], color=frozen, width=1.0, head=8.7)
    line(ax, [(6.46, 2.62), (6.46, 1.84)], color=frozen, width=1.0, head=8.7)
    junction(ax, 4.99, 3.00, frozen)
    junction(ax, 4.99, 2.62, frozen)
    label(ax, 2.06, 2.44, r'Future $A$', size=11.2, color=frozen, white=True)
    label(ax, 6.46, 2.44, r'Future $A$', size=11.2, color=frozen, white=True)

    # A separate p branch controls calls. The full Z arrow never enters or
    # terminates at the scalar gate; always-on still computes and uses Risk.
    box(ax, 3.53, 1.98, 2.00, .48, 'Invocation gate', 'Call / zero request',
        'shared', title_size=12.5, body_size=10.5)
    line(ax, [(2.95, 1.84), (2.95, 2.22), (3.53, 2.22)], color=gray,
         width=1.0, dashed=True, head=8.5)
    label(ax, 3.21, 2.36, r'$p$', size=12.2, color=gray, white=True)
    line(ax, [(5.53, 2.22), (5.80, 2.22), (5.80, 1.84)], color=gray,
         width=1.0, dashed=True, head=8.5)

    # Residual receives the same measured history in addition to A and full Z.
    line(ax, [(.84, 1.10), (.84, .82), (6.08, .82), (6.08, 1.10)],
         color=gray, width=1.15, head=8.7)
    label(ax, 3.77, .82, r'History $S$', size=11.2, color=gray, white=True)

    # The filter reads A for feasible nominal-relative arm offsets, then adds
    # the limited delta upstream of frozen SONIC. It is a slew limiter, not an
    # invented learned smoothing/low-pass stage.
    line(ax, [(7.43, 3.00), (7.43, 1.84)], color=amber, width=1.1, head=9)
    junction(ax, 7.43, 3.00, amber)
    label(ax, 7.28, 2.44, r'$A$', size=12.0, color=amber, white=True)
    line(ax, [(8.04, 1.84), (8.04, 2.875)], color=purple, width=1.7, head=10)
    label(ax, 8.19, 2.41, r'$\delta$', size=12.5, color=purple, white=True)
    label(ax, 7.90, .82, 'Zero request ramps down', size=10.5, color=amber,
          white=True)

    # Close the physical loop. The history builder reads actual execution;
    # neither labels nor future realized states enter either model.
    line(ax, [(10.30, 3.00), (10.41, 3.00), (10.41, .25), (.08, .25),
              (.08, 1.47), (.20, 1.47)], color=green, width=1.05,
         dashed=True, head=8.5)
    label(ax, 5.22, .25, 'Measured robot / object feedback', size=11.3,
          color=green, white=True)
    label(ax, 4.56, .51, r'Selective and always-on both retain full predicted $Z$.',
          size=11.3, color=teal, white=True)

    outputs = []
    for suffix in ('pdf', 'png', 'svg'):
        path = FIGURES / f'architecture.{suffix}'
        fig.savefig(path, dpi=300, facecolor='white')
        alias = FIGURES / f'risk_residual_architecture.{suffix}'
        shutil.copy2(path, alias)
        outputs.extend((path, alias))
    plt.close(fig)
    provenance = {
        'schema': 'report-horizontal-architecture-v2',
        'new_physics': False,
        'new_inference': False,
        'new_training': False,
        'source_script': str(Path(__file__).relative_to(ROOT)),
        'source_script_sha256': sha256(Path(__file__)),
        'size_inches': list(SIZE),
        'topology_authority': 'Repository README Architecture mermaid and preserved deployed controller source.',
        'controller_source_pins': {
            'risk_residual/models.py': '4e6ade62904e781818115bcb43ff4882c48e6f4984f9046618b561b1d74d922c',
            'risk_residual/runtime.py': 'd08b602d305c1e674f2ea9e69d03228004953409737e9d1c3aad4f2e857e2278',
            'baseline/reference_checks.py': '5567c40c88b635e20eba27eb2c50838e3f6117b1703f9a43c27a2a254d07ad6b',
        },
        'core_interface': 'Complete predicted body-by-future-time Z directly conditions Residual through projected additive future-token conditioning.',
        'nominal_input': 'Uncorrected future A branches upstream of SONIC into both Risk and Residual.',
        'history_input': 'Measured execution history S enters both Risk and Residual.',
        'gate': 'The scalar p schedules Residual calls; both selective and always-on preserve Risk and full Z. Low gate requests zero.',
        'shared_filter': 'Arm mask, nominal-relative joint feasibility, magnitude and slew limits precede A + delta and frozen SONIC. No extra learned or low-pass filter is depicted.',
        'finger_control': 'Shared task-driven articulated-finger control bypasses the body reference/SONIC path.',
        'source_context': 'Phase/finger context C is described in the manuscript and omitted from the concise diagram, matching README topology.',
        'future_work_only': 'No GR00T or real-robot deployment is depicted.',
        'grid': 'Schematic four body groups by eight future slots, with no activation values represented.',
        'outputs': {str(p.relative_to(ROOT)): sha256(p) for p in outputs},
    }
    (FIGURES / 'horizontal-architecture-provenance.json').write_text(
        json.dumps(provenance, indent=2) + '\n')
    print('Created README-aligned architecture: PDF, PNG and SVG (10.5 × 3.5 in).')


if __name__ == '__main__':
    draw()
