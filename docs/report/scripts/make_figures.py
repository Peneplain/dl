"""Rebuild report charts and diagrams from public inputs."""
from pathlib import Path
import subprocess
import sys

SCRIPTS = Path(__file__).resolve().parent

if __name__ == '__main__':
    for name in ['draw_horizontal_system.py', 'draw_horizontal_architecture.py',
                 'make_result_figures.py']:
        subprocess.run([sys.executable, str(SCRIPTS / name)], check=True)
