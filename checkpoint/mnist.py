"""Train the course MNIST U-Net from scratch and evaluate classifier-feature FID."""
import argparse
import copy
import csv
import math
import time
import types
from pathlib import Path

from .common import (ROOT, cell_source, frechet, gaussian, manifest, restore_rng,
                     rng_state, save_json, save_torch, setup)


def lab_module():
    # Execute the provided helper cell in memory; no generated files or network calls.
    module = types.ModuleType('mnist_guidance_lab')
    source = cell_source(ROOT / 'dsc180-d32-mnist-guidance-lab/guidance_mnist.ipynb', 3)
    exec(compile(source.split('\n', 1)[1], '<course MNIST helper>', 'exec'), module.__dict__)
    return module


def denoising_loss(net, clean, labels, dropout=0.1):
    import torch
    # Uniform time on the lab's cosine path. Clean-image MSE (not EDM loss).
    t = torch.rand(len(clean), device=clean.device)
    alpha, sigma = torch.sin(t * math.pi / 2), torch.cos(t * math.pi / 2)
    noisy = alpha[:, None, None, None] * clean + sigma[:, None, None, None] * torch.randn_like(clean)
    hidden = torch.where(torch.rand(len(labels), device=clean.device) < dropout, 10, labels)
    return (net(noisy, t, hidden) - clean).square().mean()


def train(args, mg, device):
    import torch
    from torchvision.datasets import MNIST
    out = Path(args.out)
    dataset = MNIST(ROOT / 'data', train=True, download=True)
    clean = dataset.data[:, None].float().div(127.5).sub(1)
    labels = dataset.targets
    torch.manual_seed(args.seed)
    net = mg.UNet(ch=64).to(device)  # Fresh random initialization; never load generator.pt.
    ema = copy.deepcopy(net).eval().requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    losses, start = [], 0
    latest = out / 'latest.pt'
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        for key in ['seed', 'batch', 'lr']:
            if state['config'][key] != getattr(args, key):
                raise ValueError(f'Resume changes {key}; use a different --out directory')
        net.load_state_dict(state['net'])
        ema.load_state_dict(state['ema'])
        opt.load_state_dict(state['optimizer'])
        losses, start = state['losses'], state['step']
        restore_rng(state['rng'])
        print(f'Resuming MNIST at update {start}', flush=True)
    net.train()
    started = time.time()

    def persist(step):
        save_torch(latest, dict(net=net.state_dict(), ema=ema.state_dict(), optimizer=opt.state_dict(),
                               step=step, losses=losses, rng=rng_state(), config=vars(args)))
        save_json(out / 'training.json', dict(updates=step, images_seen=step * args.batch, losses=losses,
                                            objective='uniform cosine-time clean-image MSE', label_dropout=0.1))

    try:
        for step in range(start + 1, args.updates + 1):
            idx = torch.randint(len(clean), (args.batch,))
            loss = denoising_loss(net, clean[idx].to(device), labels[idx].to(device))
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite MNIST training loss')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            opt.step()
            with torch.no_grad():
                decay = 0.9 if step <= 1000 else 0.999
                for average, current in zip(ema.parameters(), net.parameters()):
                    average.lerp_(current, 1 - decay)
            losses.append(float(loss.detach()))
            if step == start + 20 or step % 100 == 0:
                seconds = (time.time() - started) / (step - start)
                print(f'MNIST update {step}/{args.updates}, loss {sum(losses[-100:])/len(losses[-100:]):.4f}, '
                      f'{seconds:.2f} s/update, remaining training {(args.updates-step)*seconds/60:.1f} min', flush=True)
            if step % args.save_every == 0 or step == args.updates:
                persist(step)
            if step % args.snapshot_every == 0 or step == args.updates:
                save_torch(out / f'ema_{step:07d}.pt', dict(state=ema.state_dict(), ch=64, step=step))
    except KeyboardInterrupt:
        # Save the last completed update, not a partially applied update.
        print('Interrupted: resume from the most recent periodic latest.pt', flush=True)
        raise
    if not latest.exists():
        raise ValueError('No checkpoint available; use --updates > 0')
    return ema


def evaluate(args, mg, net, device):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torchvision.datasets import MNIST
    out = Path(args.out)
    evaluator = mg.Evaluator().to(device).eval().requires_grad_(False)
    weight_path = ROOT / 'dsc180-d32-mnist-guidance-lab/evaluator.pt'
    import hashlib
    if hashlib.sha256(weight_path.read_bytes()).hexdigest() != '05e955d1c14856991785c408f9a818b5d887cdc2c5142d63eac134367f50b7fe':
        raise RuntimeError('MNIST evaluator checksum mismatch')
    evaluator.load_state_dict(torch.load(weight_path, map_location='cpu', weights_only=True)['state'])
    real = MNIST(ROOT / 'data', train=False, download=True)
    real_images = real.data[:, None].float().div(127.5).sub(1)

    @torch.no_grad()
    def describe(images):
        predictions, features = [], []
        for chunk in images.split(args.eval_batch):
            logits, feats = evaluator(chunk.to(device))
            predictions.append(logits.argmax(1).cpu())
            features.append(feats.cpu())
        return torch.cat(predictions), torch.cat(features)

    predictions, real_features = describe(real_images)
    ref = gaussian(real_features)
    print(f'MNIST frozen classifier test accuracy: {(predictions == real.targets).float().mean():.2%}', flush=True)
    requested = torch.arange(args.n_eval) % 10
    results, grids = {}, {}
    net.eval().requires_grad_(False)
    for w in [0, 1, 2, 4]:
        images, nfe = mg.sample(net, args.n_eval, requested, str(device), steps=args.sample_steps,
                               w=w, seed=1000, batch=args.eval_batch)
        predicted, feats = describe(images)
        result = dict(accuracy=float((predicted == requested).float().mean()),
                      fid=frechet(*gaussian(feats), *ref), nfe=nfe,
                      pixel_spread=float(np.mean([images[requested == c].var(0, unbiased=False).mean().item()
                                                 for c in range(10)])))
        results[str(w)] = result
        grids[f'w={w}, match={result["accuracy"]:.1%}'] = images
        save_torch(out / f'samples_w{w}.pt', dict(images=images, requested=requested))
        save_json(out / 'metrics.json', dict(n_images=args.n_eval, n_reference=10000,
                  reference='MNIST test split', feature_space='frozen lab digit classifier, 128 features',
                  training_updates=args.trained_updates, results=results))
        print(f'MNIST w={w}: {result}', flush=True)
    old_show = plt.show
    def save_grid():
        plt.savefig(out / 'cfg_grid.png', dpi=180, bbox_inches='tight')
        plt.close('all')
    plt.show = save_grid
    mg.grid(grids)
    plt.show = old_show
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, key, label in zip(axes, ['accuracy', 'fid'], ['Label accuracy', 'Classifier-feature FID']):
        ax.plot([0, 1, 2, 4], [results[str(w)][key] for w in [0, 1, 2, 4]], 'o-')
        ax.set(xlabel='Guidance w', ylabel=label)
    fig.tight_layout()
    fig.savefig(out / 'cfg_metrics.png', dpi=180)
    plt.close(fig)
    with open(out / 'metrics.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['w', 'accuracy', 'fid', 'nfe', 'pixel_spread'])
        writer.writeheader()
        writer.writerows(dict(w=w, **row) for w, row in results.items())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='artifacts/mnist')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--updates', type=int, default=10000)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--eval-batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--snapshot-every', type=int, default=2000)
    parser.add_argument('--n-eval', type=int, default=1000)
    parser.add_argument('--sample-steps', type=int, default=32)
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    if min(args.updates, args.batch, args.eval_batch, args.save_every, args.snapshot_every, args.sample_steps) < 1 or args.n_eval < 20 or args.n_eval % 10:
        parser.error('Counts must be positive; n-eval must be a multiple of ten and at least twenty')
    import matplotlib
    matplotlib.use('Agg')
    import torch
    device = setup(args.device)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    manifest_name = 'evaluation_manifest.json' if args.evaluate_only else 'manifest.json'
    save_json(Path(args.out) / manifest_name, manifest(args, device))
    mg = lab_module()
    if not args.evaluate_only:
        train(args, mg, device)
    state = torch.load(Path(args.out) / 'latest.pt', map_location=device, weights_only=False)
    net = mg.UNet(ch=64).to(device)
    net.load_state_dict(state['ema'])
    args.trained_updates = state['step']
    print(f'Evaluating MNIST EMA at update {args.trained_updates}', flush=True)
    import matplotlib.pyplot as plt
    import numpy as np
    loss = state['losses']
    k = min(100, len(loss))
    plt.plot(range(k, len(loss) + 1), np.convolve(loss, np.ones(k)/k, mode='valid'))
    plt.xlabel('Update'); plt.ylabel(f'MSE ({k}-update mean)')
    plt.savefig(Path(args.out) / 'loss.png', dpi=180, bbox_inches='tight'); plt.close()
    evaluate(args, mg, net, device)


if __name__ == '__main__':
    main()
