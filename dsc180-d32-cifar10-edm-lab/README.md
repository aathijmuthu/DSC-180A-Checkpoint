# DSC 180 · D32 · EDM on CIFAR-10: conditioning, CFG and FID

One notebook, `cifar10_edm_lab.ipynb`:
- **Parts 1–5:** generate CIFAR-10 images with NVIDIA's pretrained EDM models, and measure them.
- **Part 6:** train EDM's network yourself, for CFG.

## Checkpoint assignment, due Friday, November 6

Everything in the notebook is part of it.

**In the next 24 hours**
- Create your public GitHub repository. Give it a README with the setup, the data downloads and the command
  behind each figure and table, and add a notes file.
- Get the notebook running on a GPU: the downloads and parts 1–3 take a few minutes.

**This week: generation and sampling (parts 1–5)**
- the sample grids
- the trajectory figures, with your answers to part 3's questions
- the table and plots of accuracy, FID and FID_class against w, at 1,000 images

**By the checkpoint: training (the rest)**
- Part 6: FID across snapshots for 100 and 1,000 training images, samples next to their nearest training images,
  and the CFG grids.
- An MNIST CFG network, trained with the MNIST guidance lab: label accuracy and FID for each w.
- Encouraged: an actual training run of EDM's full network on all 50,000 images, on a DSMLP GPU for as many
  hours as you can (the end of part 6). Show its loss curve and FID across snapshots.

**Submit by Friday, November 6**
- **Required by the course:** a report with the title, the abstract and the introduction (the problem, related
  work and your data), and a link to your repository.
- **Also asked by us:** a short preliminary-results section in the report, with this week's figures and tables,
  each with a sentence on what it shows. Your README should reproduce each one.

Full submission instructions and grading details: the
[official Q1 assignment](https://dsc-capstone.org/2026-27/assignments/projects/q1/).

## Run

Open the notebook on a GPU (a DSMLP GPU, or Colab) and run all cells. It needs Python 3.10–3.13 with `torch`,
`numpy` and `matplotlib`.

- **On an NVIDIA GPU,** `QUICK` is `False`. Part 5 generates 1,000 images for each value of w, and part 6 trains
  for 20,000 updates on each training set. Each long step prints how long it will take.
- **On a laptop,** `QUICK` is `True` and every step is cut down far enough to test the code, not to give results.
- **Your own EDM clone:** to use a clone of NVlabs/edm instead of the download, set the environment variable
  `EDM_DIR` to its folder.

On its first run the notebook downloads about 720 MB into `downloads/` and checks every file's checksum:
- EDM's code and its two pretrained CIFAR-10 networks
- EDM's FID reference statistics
- a pretrained CIFAR-10 classifier
- CIFAR-10 itself, from Hugging Face's copy (`uoft-cs/cifar10`), which is much faster than the original Toronto
  server. If `downloads/` already holds the original `cifar-10-python.tar.gz`, that file is used instead.

Generated images and part 6's snapshots are saved in `outputs/`.

## What it covers

1. **Unconditional samples** from EDM's pretrained network, and **inside the sampler:** a trajectory figure with
   the noisy image and the denoiser's estimate D(x; σ) at each noise level.
2. **Conditional samples,** one row per class, from the same starting noise.
3. **CFG with two networks.** EDM's conditional network was trained without a "no label" input: its
   `label_dropout` is 0. The unconditional estimate therefore comes from the separate unconditional network.
   You then make the trajectory figure for the conditional model and for CFG at w = 4.
4. **Label accuracy and FID:**
   - the FID formula and a picture of it in 2D
   - FID ignores labels, so conditional runs are scored with balanced classes
   - FID_class scores each class against real images of that class
5. **The sweep over w = 0, 1, 1.5, 2, 3, 4:** label accuracy, FID and FID_class.
6. **Training EDM's network on 100 and on 1,000 images, with label dropout 0.1:**
   - FID across saved snapshots
   - samples next to their nearest training images
   - CFG with this one network

   Encouraged: an actual training run of EDM's full network on all 50,000 images with EDM's `train.py`. The
   notebook writes the dataset, prints the command, and plots loss and FID across the run's snapshots.

## Two cautions

- **FID gets better (lower) with more samples.** With 1,000 samples it is much higher than published values,
  which use 50,000. Compare runs only with each other, at the same number of samples.
- **With few training images (part 6), the model memorizes.** Its samples are copies of the training images, so
  these runs test the training pipeline. To see new images, add more data.

## Files

- `cifar10_edm_lab.ipynb`, the notebook.
- Created when you run it:
  - `downloads/`: code, models and data
  - `outputs/`: generated images, the sweep's numbers and part 6's snapshots
