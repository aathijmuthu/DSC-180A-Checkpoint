"""Shared experiment persistence and numerical evaluation."""
import ast
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell_source(path, index):
    return ''.join(json.loads(Path(path).read_text())['cells'][index]['source'])


def definitions(source):
    tree = ast.parse(source)
    tree.body = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
    return compile(tree, '<course notebook definitions>', 'exec')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save_torch(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def manifest(args, device):
    import torch
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    except subprocess.CalledProcessError:
        commit = 'uncommitted'
    return dict(arguments=vars(args), device=str(device), gpu=torch.cuda.get_device_name(0) if device.type == 'cuda' else None,
                torch=torch.__version__, python=sys.version, platform=platform.platform(), git_commit=commit,
                notebook_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in ROOT.glob('dsc180-*/**/*.ipynb')})


def setup(device_name):
    import torch
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device(device_name if device_name != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    return device


def rng_state():
    import torch
    return dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    import torch
    torch.set_rng_state(state['cpu'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def gaussian(features):
    import torch
    if len(features) < 2:
        raise ValueError('FID needs at least two samples')
    f = features.double().cpu()
    return f.mean(0), torch.cov(f.T)


def frechet(mu1, cov1, mu2, cov2):
    """PSD sandwich avoids a nonsymmetric covariance product and complex roots."""
    import torch
    values, vectors = torch.linalg.eigh((cov1 + cov1.T) / 2)
    root = (vectors * values.clamp_min(0).sqrt()) @ vectors.T
    middle = root @ cov2 @ root
    trace_root = torch.linalg.eigvalsh((middle + middle.T) / 2).clamp_min(0).sqrt().sum()
    return max(0., float((mu1 - mu2).square().sum() + cov1.trace() + cov2.trace() - 2 * trace_root))
