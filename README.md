# DSC 180A checkpoint: EDM and classifier-free guidance

Individual project by Aathi Muthu, based on Karras et al., [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364), [NVlabs/edm](https://github.com/NVlabs/edm), and the supplied D32 labs.

[Report PDF](report/main.pdf) · [Results](RESULTS.md) · [Predictions and observations](notes.md)

Completed notebook answers: [CIFAR lab](dsc180-d32-cifar10-edm-lab/cifar10_edm_lab.ipynb) and [MNIST guidance lab, Parts 1–4](dsc180-d32-mnist-guidance-lab/guidance_mnist.ipynb). The latter includes executed comparisons using the supplied frozen generator; the checkpoint's MNIST model is trained separately from scratch. `part2_mnist.ipynb` is an optional course reference; E7 was not run.

All required experiments are complete: the six-weight CIFAR-10 sweep, two 20,000-update CIFAR training runs, and MNIST training and evaluation. The encouraged full-size EDM run was prepared but not run.

## Setup and data

Use Python 3.10–3.13 inside a DSMLP session with an allocated NVIDIA GPU:

```bash
git clone https://github.com/aathijmuthu/DSC-180A-Checkpoint.git
cd DSC-180A-Checkpoint
python -m pip install -r requirements.txt
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
bash scripts/run_dsmlp.sh all
```

The launcher runs all required experiments, writes logs to `logs/`, and records the environment under `artifacts/`. Keep the GPU allocation active. CIFAR-only and MNIST-only runs use `bash scripts/run_dsmlp.sh cifar` and `bash scripts/run_dsmlp.sh mnist`.

- **CIFAR-10:** The runner downloads the data, pretrained EDM networks, frozen ResNet-56 evaluator, Inception detector/reference statistics, and pinned EDM source to `downloads/` (about 720 MB). The course downloader verifies checksums.
- **MNIST:** Torchvision downloads the 60,000 training and 10,000 test digits to `data/`. Training starts from random weights; evaluation uses the supplied frozen `dsc180-d32-mnist-guidance-lab/evaluator.pt`.
- Keep both course lab directories: the runners reuse notebook definitions. Downloads and new model checkpoints are ignored by Git. Recorded environments and run arguments are in `artifacts/environment-*.txt` and the experiment manifests.

## Reproduce figures and tables

Run commands from the repository root. Output paths below are relative to `artifacts/`.

| Command | Outputs |
|---|---|
| `python -m checkpoint.cifar figures` | `cifar/unconditional_grid.png`, `conditional_grid.png`, `cfg_grid.png`, `unconditional_trajectory.png`, `trajectory_w1.png`, `trajectory_w4.png`; supplementary `fid_2d.png` |
| `python -m checkpoint.cifar sweep` | `cifar/cfg_sweep_n1000.json`, `cfg_sweep.csv`, `cfg_metrics.png` |
| `python -m checkpoint.cifar train100` | `cifar/subset100_training.png` (loss/FID), `subset100_nearest.png` and `.json`, `subset100_cfg.png`, `subset100_training.json`; supplementary `subset100_training_2.png` |
| `python -m checkpoint.cifar train1000` | Same outputs with prefix `subset1000` |
| `python -m checkpoint.mnist` | `mnist/cfg_grid.png`, `cfg_metrics.png`, `metrics.json`, `metrics.csv`; supplementary `loss.png` and checkpoints |
| `python -m checkpoint.mnist_lab` | Executes the frozen-generator MNIST lab on CPU and saves four grids and metric outputs inside `guidance_mnist.ipynb`; no FID for its 40-image comparisons |
| `python -m checkpoint.summarize` | `RESULTS.md` and the four report tables in `report/results.tex` |

Training resumes from `latest.pt`; the CIFAR sweep reuses cached results. For a fresh checkpoint experiment, append `--out artifacts/reproduction-cifar` to each CIFAR command or `--out artifacts/reproduction-mnist` to `python -m checkpoint.mnist`. Keep sampling batch size and other settings fixed across paired comparisons. Report tables read the default output directories.

## Experiment settings

- **Pretrained CIFAR:** 18 Heun steps; 1,000 balanced samples per weight at `w = 0, 1, 1.5, 2, 3, 4`; paired noise; sampling batch 32. Global FID uses NVIDIA's 50,000-image reference. Class FID averages ten comparisons of 100 generated images against 5,000 real training images per class; it is omitted at `w=0`, which ignores requests.
- **Restricted-data CIFAR:** Balanced subsets of 100 and 1,000 images; 7M-parameter EDM network; 10% label dropout; EDMLoss; Adam at 2e-4; batch 64; EMA; 20,000 updates. Five snapshots each receive a 1,000-image unconditional FID evaluation. Nearest-training-image comparisons use 100 generated and 100 real test images.
- **MNIST:** Fresh lab U-Net, all 60,000 training digits, 10% null labels, uniform cosine time and clean-image MSE, Adam at 2e-4, batch 128, EMA, 10,000 updates. Evaluation uses 32 DDIM steps and 1,000 balanced samples per weight at `w = 0, 1, 2, 4`. FID uses the frozen classifier's 128 features against all 10,000 test digits, so its scale differs from CIFAR Inception FID.

The report and notes document the interrupted MNIST run and the CIFAR transfer from Mac to DSMLP. These are single-seed experiments; small FID differences are not significance claims.

## Build and check

Rebuild the PDF from saved results; LaTeX is required:

```bash
python -m checkpoint.summarize
(cd report && pdflatex -interaction=nonstopmode -halt-on-error main.tex && pdflatex -interaction=nonstopmode -halt-on-error main.tex)
python -m unittest discover -s tests -v
```

The [Colab runner](https://colab.research.google.com/github/aathijmuthu/DSC-180A-Checkpoint/blob/main/colab_checkpoint.ipynb) is a CIFAR GPU fallback. Optional full-data preparation: `python -m checkpoint.prepare_full_edm` exports the 50,000-image dataset and writes a launch script without starting training.

## AI use

I used OpenAI Codex for development, debugging, experiment automation, and writing assistance. My understanding of the methods comes from reading the EDM paper myself.
