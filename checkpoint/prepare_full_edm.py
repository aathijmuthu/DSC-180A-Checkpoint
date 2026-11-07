"""Export the full CIFAR dataset and print the optional CUDA training command."""
import argparse
import io
import json
import pickle
import shlex
import tarfile
import zipfile
from pathlib import Path

from .common import ROOT, cell_source, definitions, save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='data/edm')
    args = parser.parse_args()
    import numpy as np
    import torch
    ns = dict(torch=torch, np=np, io=io, pickle=pickle, tarfile=tarfile, zipfile=zipfile, json=json,
              DL=ROOT / 'downloads', TAR=ROOT / 'downloads/cifar-10-python.tar.gz')
    notebook = ROOT / 'dsc180-d32-cifar10-edm-lab/cifar10_edm_lab.ipynb'
    exec(definitions(cell_source(notebook, 5)), ns)
    images, labels = ns['load_cifar10']('train')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    ns.update(OUT=out, train_images=images, train_labels=labels)
    exec(definitions(cell_source(notebook, 37)), ns)
    dataset = out / 'cifar10-32x32-n50000.zip'
    if not dataset.exists():
        print('Exporting all 50,000 CIFAR-10 training images in EDM format', flush=True)
        ns['export_edm_dataset']()
    # Verify the archive before presenting a command that consumes it.
    with zipfile.ZipFile(dataset) as archive:
        metadata = json.loads(archive.read('dataset.json'))
        assert len(metadata['labels']) == 50000
        assert len(archive.namelist()) == 50001
        counts = [sum(label == c for _, label in metadata['labels']) for c in range(10)]
        assert counts == [5000] * 10
        assert archive.testzip() is None
    edm_dir = ROOT / 'downloads/edm-008a4e5316c8e3bfe61a62f874bddba254295afb'
    command = ['torchrun', '--standalone', '--nproc_per_node=1', 'train.py', '--outdir=training-runs',
               f'--data={dataset}', '--cond=0', '--arch=ddpmpp', '--batch=512', '--batch-gpu=16',
               '--duration=10', '--snap=10', '--dump=10']
    script = '#!/usr/bin/env bash\nset -euo pipefail\n'
    script += 'python -c ' + shlex.quote('import torch; assert torch.cuda.is_available(), "This optional run needs a CUDA GPU"') + '\n'
    script += 'cd ' + shlex.quote(str(edm_dir)) + '\n' + shlex.join(command) + '\n'
    (out / 'train_full_edm.sh').write_text(script)
    save_json(out / 'dataset_manifest.json', dict(n_images=50000, class_counts=counts,
              format='EDM ZIP_STORED PNGs and dataset.json', source='checksummed CIFAR-10 lab download',
              training='optional full-size unconditional DDPM++ network; not launched'))
    print(f'Verified dataset: {dataset}')
    print(f'Optional launch script: {out / "train_full_edm.sh"}')
    print('Run this preparation command again on DSMLP to generate paths valid on that machine.')
    print('The optional training run has not been started.')


if __name__ == '__main__':
    main()
