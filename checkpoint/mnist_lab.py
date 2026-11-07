"""Execute the supplied MNIST guidance exercises and save real notebook outputs."""
import argparse
import base64
import contextlib
import io
import json
import os
import sys
from pathlib import Path

from .common import ROOT
from .mnist import lab_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'mps'])
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = ROOT / 'dsc180-d32-mnist-guidance-lab'
    path = folder / 'guidance_mnist.ipynb'
    notebook = json.loads(path.read_text())
    # The notebook's helper is executed in memory instead of writing a .py file.
    sys.modules['mnist_guidance_lab'] = lab_module()
    namespace = {'__name__': '__mnist_lab__'}
    previous_directory = Path.cwd()
    previous_device = os.environ.get('LAB_DEVICE')
    os.environ['LAB_DEVICE'] = args.device
    try:
        os.chdir(folder)
        for count, index in enumerate((4, 5, 7, 9, 11), 1):
            cell = notebook['cells'][index]
            outputs, stdout = [], io.StringIO()

            def show():
                for number in plt.get_fignums():
                    figure = plt.figure(number)
                    buffer = io.BytesIO()
                    figure.savefig(buffer, format='png', dpi=120, bbox_inches='tight')
                    outputs.append(dict(output_type='display_data', metadata={}, data={
                        'image/png': base64.b64encode(buffer.getvalue()).decode('ascii'),
                        'text/plain': ['<MNIST guidance comparison figure>']}))
                    plt.close(figure)

            previous_show = plt.show
            plt.show = show
            try:
                with contextlib.redirect_stdout(stdout):
                    exec(compile(''.join(cell['source']), f'{path.name}:cell{index}', 'exec'), namespace)
            finally:
                plt.show = previous_show
            if stdout.getvalue():
                outputs.insert(0, dict(output_type='stream', name='stdout', text=stdout.getvalue().splitlines(True)))
            cell.update(execution_count=count, outputs=outputs)
            # Persist only executed cells; never label a failed or skipped cell as run.
            path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
            print(f'Completed MNIST guidance cell {index} on {args.device}', flush=True)
            print(stdout.getvalue(), flush=True)
    finally:
        os.chdir(previous_directory)
        if previous_device is None:
            os.environ.pop('LAB_DEVICE', None)
        else:
            os.environ['LAB_DEVICE'] = previous_device


if __name__ == '__main__':
    main()
