# DSC 180A checkpoint: EDM, conditioning, and guidance

Individual project by Aathi Muthu, based on Karras et al., [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) (NeurIPS 2022), [NVlabs/edm](https://github.com/NVlabs/edm), and the supplied D32 labs.

## Start on DSMLP

Open a terminal **inside your GPU-enabled DSMLP session**. Clone this repository (or upload the project folder), then run:

```bash
cd DSC-180A-Checkpoint
python --version
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m pip install -r requirements.txt
bash scripts/run_dsmlp.sh all
```

Use Python 3.10–3.13. If your DSMLP image already includes CUDA-enabled PyTorch, the requirements accept compatible installed versions. Verify CUDA again after installation. A login terminal without an allocated GPU is insufficient. The launcher checks for CUDA before starting and records the environment in `artifacts/environment.txt`.

Keep the GPU session alive for the run; `tmux` can preserve the terminal if available, but does not extend the GPU allocation. Logs appear in `logs/`. The launcher runs jobs in sequence on one GPU and continues to other deliverables if a job fails; it exits unsuccessfully if any experiment failed. Inspect the log named by the failure message.

If you have **two separately allocated GPUs/sessions**, start these in separate terminals with the appropriate GPU visible in each:

```bash
bash scripts/run_dsmlp.sh cifar
bash scripts/run_dsmlp.sh mnist
```

Do not launch two training jobs on the same GPU until memory capacity is established. Both tracks are independent. Re-running a training command resumes `latest.pt`. CIFAR pretrained samples are cached. Downloads and large checkpoints are ignored by Git.

## Individual experiments and figure commands

Run from the repository root. No notebook execution is required; the runners read the supplied notebooks to reuse their implementations. **Keep the course notebook files in the repository.**

| Command | Figures and tables under `artifacts/` |
|---|---|
| `python -m checkpoint.cifar figures` | `cifar/unconditional_grid.png`, `conditional_grid.png`, `cfg_grid.png`, `unconditional_trajectory.png`, `trajectory_w1.png`, `trajectory_w4.png`, `fid_2d.png` |
| `python -m checkpoint.cifar sweep` | `cifar/cfg_sweep_n1000.json`, `cfg_sweep.csv`, `cfg_metrics.png` |
| `python -m checkpoint.cifar train100` | `cifar/subset100_training.png` (loss/FID), `subset100_training_2.png` (snapshot grids), `subset100_nearest.png`, `subset100_nearest.json`, `subset100_cfg.png`, `subset100_training.json` |
| `python -m checkpoint.cifar train1000` | Equivalent files with prefix `subset1000` |
| `python -m checkpoint.mnist` | `mnist/loss.png`, `cfg_grid.png`, `cfg_metrics.png`, `metrics.csv`, `metrics.json`, EMA checkpoints and generated samples |
| `python -m checkpoint.mnist --evaluate-only` | Re-evaluate the saved MNIST EMA without further training |
| `python -m checkpoint.summarize` | `RESULTS.md` and `report/results.tex`, populated only from available metrics |

Figures also enter the report automatically when their files exist. Interpretation questions and predictions are in [notes.md](notes.md). [RESULTS.md](RESULTS.md) records the results currently available. **A successful local smoke check is not a completed assignment experiment.**

## Data and model downloads

- **CIFAR-10:** The original lab downloader writes to `downloads/`, verifies its fixed SHA-256 checksums, and fetches a pinned Hugging Face copy of the training/test data, NVIDIA's conditional/unconditional pretrained models and Inception detector/reference statistics, a pretrained ResNet-56 evaluator, and NVlabs/edm revision `008a4e5316c8e3bfe61a62f874bddba254295afb`. Budget about 720 MB for the initial downloads plus checkpoint/sample storage. If `downloads/cifar-10-python.tar.gz` exists, the lab uses that original archive. The lab applies its PyTorch compatibility fix only to its downloaded EDM copy. An existing checkout can be selected with `EDM_DIR`.
- **MNIST:** `torchvision.datasets.MNIST` downloads the 60,000 training and 10,000 test images to `data/`. Training uses **fresh random initialization**, not `generator.pt`. The provided `dsc180-d32-mnist-guidance-lab/evaluator.pt` is the frozen evaluator; its SHA-256 is verified. The other supplied weights remain as course reference assets.

The two original MNIST notebooks are reference labs. Their earlier classroom exercises are distinct from this checkpoint's MNIST training requirement. The optional unconditional MNIST notebook is not part of the required experiment runner.

## Protocol

CIFAR pretrained evaluation uses 1,000 images at each `w = 0, 1, 1.5, 2, 3, 4`, balanced requested classes, 18 deterministic Heun steps, and paired CPU-generated noise. Sampling/evaluation batch size defaults to 32 to reduce GPU memory use. Keep batch size fixed across comparisons because the CIFAR lab seeds each chunk separately. The runner rejects a changed cached sweep configuration.

All-class FID uses NVIDIA's 50,000-image training reference. FID_class averages ten requested-class comparisons, each with 100 generated images and 5,000 real training images. At w=0, requests are ignored and class FID is omitted. Classifier accuracy is also measured on real CIFAR test images. These small-sample FIDs are not direct replications of EDM's published 50,000-sample values.

Small CIFAR training follows the notebook: balanced subsets of 100 and 1,000 images; 7M-parameter SongUNet with EDM preconditioning and EDMLoss; label dropout 0.1; Adam at 2e-4; batch 64; EMA; 20,000 updates per subset; five snapshot evaluations of 1,000 unconditional samples each. Pixel-nearest neighbors are compared against a real-test-image baseline. Full-data 55.7M-parameter training is encouraged in the assignment but is outside the default deadline run.

MNIST uses the lab U-Net (64 base channels), the lab cosine path, uniformly sampled times, and clean-image MSE. This loss is our explicit training choice because the supplied notebook has no trainer. Training uses all 60,000 digits, label dropout 0.1 (label 10 means no request), Adam at 2e-4, batch 128, gradient clipping at norm 1, and EMA. The default is 10,000 updates; actual updates are recorded in metrics. Evaluation uses 1,000 balanced requests at `w = 0, 1, 2, 4`, paired initial noise, and 32 deterministic DDIM steps. The real reference is all 10,000 held-out MNIST test digits. **MNIST FID uses the lab classifier's 128 features, not Inception features**, so its scale is separate from CIFAR FID.

## Resuming and changing the time budget

Checkpoint saves are atomic. Training saves model, EMA, optimizer, RNG states, update count, and losses every 500 updates. CIFAR also saves at every evaluated snapshot; MNIST exports EMA snapshots every 2,000 updates. Interruptions resume from the most recent completed save. Keep the configuration fixed when resuming; use a new `--out` for changed experiment settings. Do not run two writers against the same output directory.

```bash
# Required defaults:
python -m checkpoint.cifar train100 --updates 20000
python -m checkpoint.cifar train1000 --updates 20000
python -m checkpoint.mnist --updates 10000

# Explicit shorter preliminary runs, if the GPU benchmark exceeds your remaining time:
python -m checkpoint.cifar train100 --updates 4000 --out artifacts/preliminary-cifar100
python -m checkpoint.mnist --updates 2000 --out artifacts/preliminary-mnist
```

Shorter runs must be described as partial training; they do not silently replace the default result directories. `CIFAR_UPDATES` and `MNIST_UPDATES` can override launcher schedules **before the first run**. The scripts print observed seconds/update and estimated remaining time; use that rather than assuming a runtime for an unknown GPU. Lower `--batch` for CIFAR sampling or `--eval-batch` for MNIST if evaluation runs out of memory. Lowering training batch changes the protocol and requires a separate directory and disclosure.

## Build the report

The LaTeX draft contains the title, abstract, introduction (problem, related work, data), experimental protocol, and preliminary-results slots. Missing experiments are explicitly marked as pending. **Do not submit the draft as if those experiments were completed.** After running experiments:

```bash
python -m checkpoint.summarize
cd report
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Review `report/main.pdf`, add evidence-based interpretations to `notes.md` and `report/main.tex`, and rebuild. In particular, answer the three trajectory questions, choose a guidance weight with justification, discuss nearest-neighbor evidence, and describe the MNIST tradeoff. Each figure needs a sentence about the observed result. The optional full-data training run has not been claimed or scheduled.

If DSMLP lacks LaTeX, download the `artifacts/` folder to this local project or compile with the provided source in Overleaf. To share results for review, return the JSON/CSV files, PNG figures, and logs; large model checkpoints are not needed to write the report.

## Local verification

```bash
python -m unittest discover -s tests -v
python -m compileall -q checkpoint
bash -n scripts/run_dsmlp.sh
```

An optional small end-to-end MNIST check downloads real data, performs two updates, evaluates every guidance weight, and saves figures. Its output is isolated and ignored by Git:

```bash
python -m checkpoint.mnist --out artifacts/smoke-mnist --device cpu --updates 2 --batch 2 --eval-batch 64 --n-eval 20 --sample-steps 2 --save-every 1 --snapshot-every 1
```
