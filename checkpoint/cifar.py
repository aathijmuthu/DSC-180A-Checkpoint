"""Run the supplied CIFAR EDM lab with saved figures and resumable training."""
import argparse
import ast
import copy
import csv
import fcntl
import json
import os
import time
from pathlib import Path

from .common import (ROOT, cell_source, definitions, manifest, restore_rng, rng_state,
                     save_json, save_torch, setup)

NOTEBOOK = ROOT / 'dsc180-d32-cifar10-edm-lab/cifar10_edm_lab.ipynb'


class Lab:
    def __init__(self, args):
        import matplotlib.pyplot as plt
        # Initialize the backend before replacing show. New Matplotlib versions
        # annotate pyplot.show during backend selection, which rejects bound methods.
        plt.switch_backend('Agg')
        self.args = args
        self.out = Path(args.out).resolve()
        self.out.mkdir(parents=True, exist_ok=True)
        self.ns = {'__name__': '__course_lab__'}
        self.name, self.count = 'figure', 0
        plt.show = self.save_figures

    def save_figures(self):
        import matplotlib.pyplot as plt
        for number in plt.get_fignums():
            fig = plt.figure(number)
            self.count += 1
            suffix = '' if self.count == 1 else f'_{self.count}'
            fig.savefig(self.out / f'{self.name}{suffix}.png', dpi=180, bbox_inches='tight')
            plt.close(fig)

    def cell(self, index, name=None, only_definitions=False):
        self.name, self.count = name or f'cell_{index}', 0
        source = cell_source(NOTEBOOK, index)
        # Reduce memory without changing the evaluation sample count.
        source = source.replace('range(0, len(images), 500)', 'range(0, len(images), BATCH)')
        source = source.replace('images[i:i + 500]', 'images[i:i + BATCH]')
        source = source.replace('range(0, n, 500)', 'range(0, n, BATCH)').replace('labels[b:b + 500]', 'labels[b:b + BATCH]')
        source = source.replace('images[i:i + 100]', 'images[i:i + BATCH]').replace('range(0, len(images), 100)', 'range(0, len(images), BATCH)')
        code = definitions(source) if only_definitions else compile(source, f'{NOTEBOOK.name}:cell{index}', 'exec')
        exec(code, self.ns)

    def prepare(self):
        self.cell(1)
        self.ns.update(DEVICE=self.args.device_object, QUICK=False, N_EVAL=self.args.n_eval, BATCH=self.args.batch)
        print(f'Runner configuration: device={self.args.device_object}, QUICK=False, '
              f'n_eval={self.args.n_eval}, sampling batch={self.args.batch}', flush=True)
        self.cell(3)  # checksummed downloads and pinned EDM source
        # Execute data/plot helpers while omitting pretrained loads in training-only jobs.
        source = ast.parse(cell_source(NOTEBOOK, 5))
        if self.args.task.startswith('train'):
            source.body = [n for n in source.body if not (isinstance(n, ast.Assign) and
                           any(isinstance(t, ast.Name) and t.id in ('uncond', 'cond') for t in n.targets))]
        exec(compile(source, '<course data setup>', 'exec'), self.ns)
        for index in [7, 9, 15]:
            self.cell(index, only_definitions=True)
        self.ns['W'] = [0, 1, 1.5, 2, 3, 4]
        self.cell(19)
        self.cell(21)
        self.ns['OUT'] = self.out
        save_json(self.out / f'manifest_{self.args.task}.json', manifest(self.args_for_manifest(), self.args.device_object))

    def args_for_manifest(self):
        import argparse
        return argparse.Namespace(**{k: v for k, v in vars(self.args).items() if k != 'device_object'})

    def figures(self):
        for index, name in [(7, 'unconditional_grid'), (9, 'unconditional_trajectory'),
                            (11, 'conditional_grid'), (13, 'label_dropout'), (15, 'cfg_grid'), (23, 'fid_2d')]:
            self.cell(index, name)
        ns = self.ns
        import torch
        labels = torch.full((3,), ns['CLASSES'].index('cat'))
        for w in [1, 4]:
            self.name, self.count = f'trajectory_w{w}', 0
            _, history = ns['edm_sample'](lambda x, s: ns['D_cfg'](x, s, labels, w), n=3, seed=2, record=True)
            ns['show_trajectory'](history, title=f'cat, w={w}, seed=2')
            save_torch(self.out / f'trajectory_w{w}.pt', history)

    def sweep(self):
        import torch
        import hashlib
        ns = self.ns
        protocol = dict(batch=self.args.batch, n_eval=self.args.n_eval, steps=18, seed_base=1000,
                        notebook_sha256=hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest())
        path = self.out / 'sweep_protocol.json'
        if path.exists() and json.loads(path.read_text()) != protocol:
            raise ValueError('Sampling configuration changed; use a different --out directory to avoid stale cached images')
        save_json(path, protocol)
        self.cell(26, only_definitions=True)
        # Reference moments are expensive to compute. Reuse the exact double-
        # precision moments on reruns, and persist each completed guidance weight.
        ref_path = self.out / 'class_reference.pt'
        if ref_path.exists():
            ns['CLASS_REF'] = torch.load(ref_path, weights_only=True)['moments']
            print('Reusing saved CIFAR class-reference moments', flush=True)
        else:
            print('Extracting Inception features of all 50,000 real training images for class FID', flush=True)
            chunks = []
            for start in range(0, len(ns['train_images']), 4096):
                chunks.append(ns['features'](ns['train_images'][start:start + 4096]))
                print(f'Reference features: {min(start + 4096, len(ns["train_images"]))}/{len(ns["train_images"])}', flush=True)
            real_features = torch.cat(chunks)
            del chunks
            ns['CLASS_REF'] = [ns['gaussian'](real_features[ns['train_labels'] == c]) for c in range(10)]
            save_torch(ref_path, dict(moments=ns['CLASS_REF'], n_reference=50000))
            del real_features
            print('Saved CIFAR class-reference moments', flush=True)
        result_path = self.out / f'cfg_sweep_n{self.args.n_eval}.json'
        results = {}
        if result_path.exists():
            results = {float(w): row for w, row in json.loads(result_path.read_text())['results'].items()}
        for w in ns['W']:
            if w in results:
                print(f'Reusing completed metrics for w={w:g}', flush=True)
                continue
            images, labels = ns['generate'](w)
            predicted, feats = ns['classify'](images), ns['features'](images)
            results[w] = dict(accuracy=float((predicted == labels).float().mean()),
                              fid=ns['frechet'](*ns['gaussian'](feats), ns['MU_REF'], ns['SIGMA_REF']),
                              fid_class=ns['fid_class'](feats, labels) if w > 0 else None)
            save_json(result_path, dict(n_images=self.args.n_eval, complete=len(results) == len(ns['W']),
                      results={f'{key:g}': value for key, value in results.items()}))
            print(f'CIFAR w={w:g}: {results[w]}', flush=True)
        images, _ = ns['generate'](0)
        ns.update(results=results, histogram=torch.bincount(ns['classify'](images), minlength=10))
        self.cell(27, 'cfg_metrics')
        with open(self.out / 'cfg_sweep.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['w', 'accuracy', 'fid', 'fid_class'], lineterminator='\n')
            writer.writeheader()
            writer.writerows(dict(w=w, **row) for w, row in self.ns['results'].items())

    def train(self, size):
        import numpy as np
        import torch
        ns, args = self.ns, self.args
        self.cell(29, only_definitions=True)
        ns.update(LABEL_DROPOUT=0.1, BATCH_TRAIN=args.train_batch)
        clean, labels, _ = ns['subset'](size)
        torch.manual_seed(0)
        net = ns['EDMPrecond'](img_resolution=32, img_channels=3, label_dim=10, model_type='SongUNet',
                              label_dropout=0.1, model_channels=64, channel_mult=[1, 2, 2], num_blocks=2,
                              attn_resolutions=[16], dropout=0., embedding_type='positional', encoder_type='standard',
                              decoder_type='standard', channel_mult_noise=1, resample_filter=[1, 1]).to(args.device_object).train()
        ema = copy.deepcopy(net).eval().requires_grad_(False)
        optimizer = torch.optim.Adam(net.parameters(), lr=2e-4)
        objective = ns['EDMLoss']()
        losses, fids, progress, start = [], {}, {}, 0
        latest = self.out / f'subset{size}_latest.pt'
        if latest.exists():
            state = torch.load(latest, map_location='cpu', weights_only=False)
            for key in ['updates', 'train_batch', 'batch', 'n_eval']:
                if state['config'][key] != getattr(args, key):
                    raise ValueError(f'Changing {key} on resume requires a new --out directory')
            net.load_state_dict(state['net']); ema.load_state_dict(state['ema'])
            optimizer.load_state_dict(state['optimizer'])
            losses, fids, progress, start = state['losses'], state['fids'], state['progress'], state['step']
            restore_rng(state['rng'])
            print(f'Resuming subset {size} at update {start}', flush=True)

        def persist(step):
            save_torch(latest, dict(net=net.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
                       losses=losses, fids=fids, progress=progress, step=step, rng=rng_state(),
                       config=vars(self.args_for_manifest())))
            save_json(self.out / f'subset{size}_training.json', dict(n_train=size, updates=step,
                      images_seen=step * args.train_batch, n_eval=args.n_eval, losses=losses, fids=fids))

        interval = max(1, args.updates // 5)
        started = time.monotonic()
        end = min(args.updates, args.stop_after or args.updates)
        if end < start:
            raise ValueError('--stop-after is earlier than the saved checkpoint')
        for step in range(start + 1, end + 1):
            indices = torch.randint(len(clean), (args.train_batch,))
            y = torch.nn.functional.one_hot(labels[indices], 10).float().to(args.device_object)
            loss = objective(net, clean[indices].to(args.device_object), y).mean()
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite CIFAR training loss')
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
            with torch.no_grad():
                for average, current in zip(ema.parameters(), net.parameters()):
                    average.lerp_(current, 1 - (0.9 if step <= 1000 else 0.999))
            losses.append(float(loss.detach()))
            if step == start + 20 or step % 100 == 0:
                rate = (time.monotonic() - started) / (step - start)
                print(f'CIFAR subset {size}, update {step}/{args.updates}, loss {np.mean(losses[-100:]):.3f}, '
                      f'{rate:.2f} s/update, remaining to stage update {end} '
                      f'incl. elapsed evaluation ~{rate*(end-step)/60:.1f} min', flush=True)
            if step % args.save_every == 0 or step == end:
                persist(step)
            if step % interval == 0 or step == args.updates:
                save_torch(self.out / f'subset{size}_update{step}.pt', ema.state_dict())
                images = ns['own_samples'](ema, args.n_eval)[0]
                fids[step] = ns['fid'](images)
                progress[step] = images[:10]
                print(f'CIFAR subset {size}, update {step}: FID {fids[step]:.2f}', flush=True)
                persist(step)
                # Publish useful figures even before the run finishes.
                ns.update(runs={size: dict(net=ema, losses=losses, fids=fids, progress=progress)}, N_SNAPSHOT=args.n_eval)
                self.cell(31, f'subset{size}_training')
        ns.update(runs={size: dict(net=ema, losses=losses, fids=fids, progress=progress)}, N_SNAPSHOT=args.n_eval)
        self.cell(31, f'subset{size}_training')
        # Record distances as structured evidence as well as printing notebook output.
        self.cell(33, f'subset{size}_nearest')
        samples = ns['own_samples'](ema, 100, seed=20)[0]
        d_gen, _ = ns['nearest'](samples, clean)
        d_test, _ = ns['nearest'](ns['test_images'][:100], clean)
        save_json(self.out / f'subset{size}_nearest.json', dict(n_samples=100,
                  median_generated_distance=float(d_gen.median()), median_test_distance=float(d_test.median())))
        self.cell(35, f'subset{size}_cfg')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['prepare', 'figures', 'sweep', 'train100', 'train1000'])
    parser.add_argument('--out', default='artifacts/cifar')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch', type=int, default=32, help='sampling/evaluation batch; keep fixed across paired runs')
    parser.add_argument('--train-batch', type=int, default=64)
    parser.add_argument('--n-eval', type=int, default=1000)
    parser.add_argument('--updates', type=int, default=20000)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--stop-after', type=int, help='stop at this update while retaining the full snapshot schedule for resume')
    args = parser.parse_args()
    if args.stop_after is not None and args.stop_after < 1:
        parser.error('--stop-after must be positive')
    if min(args.batch, args.train_batch, args.updates, args.save_every) < 1 or args.n_eval < 20 or args.n_eval % 10:
        parser.error('Counts must be positive; n-eval must be a multiple of ten and at least twenty')
    import matplotlib
    matplotlib.use('Agg')
    Path(args.out).mkdir(parents=True, exist_ok=True)
    # A queued continuation may reach a task while an earlier pilot is still
    # running. Serialize writers for that task before loading GPU resources.
    task_lock = open(Path(args.out) / f'.{args.task}.lock', 'a')
    fcntl.flock(task_lock, fcntl.LOCK_EX)
    args.device_object = setup(args.device)
    os.environ['LAB_DEVICE'] = str(args.device_object)
    os.chdir(ROOT)
    lab = Lab(args)
    lab.prepare()
    # Training from the lab and sampling use float32, as in the supplied notebook.
    if args.task == 'prepare':
        print('CIFAR downloads, data, pretrained models, and evaluators are ready', flush=True)
    elif args.task == 'figures': lab.figures()
    elif args.task == 'sweep': lab.sweep()
    else: lab.train(100 if args.task == 'train100' else 1000)


if __name__ == '__main__':
    main()
