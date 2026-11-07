"""Build report tables from observed metrics; missing runs remain explicitly pending."""
import json
from pathlib import Path
from .common import ROOT


def table(headers, rows):
    text = ['\\begin{center}', '\\begin{tabular}{' + 'r' * len(headers) + '}', '\\hline',
            ' & '.join(headers) + r' \\', '\\hline']
    text += [' & '.join(map(str, row)) + r' \\' for row in rows]
    return '\n'.join(text + ['\\hline', '\\end{tabular}', '\\end{center}'])


def main():
    lines = []
    md = ['# Observed experiment results', '']
    cifar = ROOT / 'artifacts/cifar/cfg_sweep_n1000.json'
    mnist = ROOT / 'artifacts/mnist/metrics.json'
    if cifar.exists():
        data = json.loads(cifar.read_text())
        if data.get('complete') is False:
            lines += ['\\paragraph{Sweep status.} This is a partial sweep; remaining guidance weights are still running.']
            md += ['CIFAR-10 guidance sweep is partially complete.', '']
        rows = [[w, f'{r["accuracy"]:.3f}', f'{r["fid"]:.2f}',
                 '--' if r['fid_class'] is None else f'{r["fid_class"]:.2f}'] for w, r in data['results'].items()]
        lines += ['\\subsection{Pretrained CIFAR-10 guidance}',
                  f'Table: {data["n_images"]} generated images per setting, balanced requests. '
                  'Class FID uses 100 generated images per class and is omitted for $w=0$.',
                  table(['$w$', 'Accuracy', 'FID', 'Class FID'], rows)]
        md += ['## CIFAR-10 guidance', '', '| w | Accuracy | FID | FID_class |', '|---|---|---|---|']
        md += ['| ' + ' | '.join(row) + ' |' for row in rows]
    else:
        lines += ['\\paragraph{CIFAR-10 guidance.} The required 1,000-image sweep has not yet been imported.']
        md += ['CIFAR-10 guidance sweep: pending.', '']
    for size in [100, 1000]:
        path = ROOT / f'artifacts/cifar/subset{size}_training.json'
        if path.exists():
            data = json.loads(path.read_text())
            lines += [f'\\subsection{{CIFAR-10 training: {size} images}}',
                      f'The saved run reached {data["updates"]:,} updates; snapshot FID uses {data["n_eval"]} generated images.',
                      table(['Update', 'FID'], [[s, f'{v:.2f}'] for s, v in data['fids'].items()])]
            md += [f'## CIFAR subset {size}: {data["updates"]} updates', '', '| Update | FID |', '|---|---|']
            md += [f'| {s} | {v:.2f} |' for s, v in data['fids'].items()]
            near = ROOT / f'artifacts/cifar/subset{size}_nearest.json'
            if near.exists():
                distances = json.loads(near.read_text())
                lines += [f'Median nearest-training-image pixel distance: generated '
                          f'{distances["median_generated_distance"]:.2f}; real test images '
                          f'{distances["median_test_distance"]:.2f}. The comparison uses 100 images per group.']
                if distances['median_generated_distance'] < distances['median_test_distance']:
                    lines += ['Generated samples are closer to the training set than real test images by this metric, '
                              'a pattern consistent with memorization. The image pairs show how closely individual samples match.']
                md += ['', f'Median nearest-training-image distance: generated '
                       f'{distances["median_generated_distance"]:.2f}; real test '
                       f'{distances["median_test_distance"]:.2f}.', '']
        else:
            lines += [f'\\paragraph{{CIFAR-10 training: {size} images.}} Results have not yet been imported.']
    if mnist.exists():
        data = json.loads(mnist.read_text())
        rows = [[w, f'{r["accuracy"]:.3f}', f'{r["fid"]:.2f}'] for w, r in data['results'].items()]
        lines += ['\\subsection{MNIST guidance}',
                  f'The model was trained from scratch for {data["training_updates"]:,} updates. '
                  f'Evaluation uses {data["n_images"]} generated images per setting and {data["n_reference"]} real test images. '
                  'FID is measured in the frozen digit classifier feature space.',
                  table(['$w$', 'Accuracy', 'Feature FID'], rows)]
        md += ['## MNIST guidance', '', '| w | Accuracy | Classifier-feature FID |', '|---|---|---|']
        md += ['| ' + ' | '.join(row) + ' |' for row in rows]
    else:
        lines += ['\\paragraph{MNIST guidance.} Training and evaluation results have not yet been imported.']
        md += ['MNIST training and evaluation: pending.', '']
    (ROOT / 'report/results.tex').write_text('\n\n'.join(lines) + '\n')
    (ROOT / 'RESULTS.md').write_text('\n'.join(md) + '\n')
    print('Updated report/results.tex and RESULTS.md from available metrics. Add visual observations to notes.md and the report.')


if __name__ == '__main__':
    main()
