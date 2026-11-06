# DSC 180 · D32 · MNIST and guidance lab

Two standalone notebooks. Each one installs its packages, writes its helper code next to itself, and fetches the
trained weights (about 10 MB, checked against the course copy), so the notebook alone is enough. This folder also
holds a copy of the weights for working offline: keep them next to the notebooks and the fetch cell skips the
download.

- `guidance_mnist.ipynb`: the class-conditional MNIST generator. Change the request, then the guidance weight,
  with the starting noise held fixed. About a minute on a laptop CPU. **This is the assignment (Parts 1–4).**
- `part2_mnist.ipynb`: the unconditional MNIST denoiser from class: the forward process on digits, what the
  denoiser predicts, the class sampler, and **E7** (when is the digit decided?). About a minute on a CPU; it uses
  a GPU if PyTorch can. Optional.

## Run

Open a notebook in Jupyter, VS Code or Colab (File → Upload notebook), run the setup cells, then
**Restart Kernel and Run All**. Python 3.10–3.13 (3.9 also works); the setup cell installs `torch`, `numpy` and
`matplotlib` if they are missing.

## Assignment: `guidance_mnist.ipynb`, Parts 1–4

1. Change the condition, keep the noise.
2. Change the guidance weight: w = 0 (no request), 1 (as trained), 2, 5.
3. Your controlled comparison: another digit and another seed, with the weights, the steps and the paired noise
   held fixed. Two labeled grids with their metric tables, and 150–250 words on the three questions.
4. Connect the code to the architecture: where the request enters `UNet.forward` (`cemb`), pixel space versus
   latent space, U-Net versus DiT, and which requests this model can and cannot express.

Optional: **E7** in `part2_mnist.ipynb`.

Write predictions before running, and keep comparisons paired: one change at a time. The digit classifier is only
an evaluator; it never guides the sampler, and its match rate can miss malformed digits. Strong guidance is not
automatically a better result.

## Grading (10 points)

Runnable comparison 2, paired controls 3, clear evidence 2, interpretation and architecture connections 3.
A negative result with a sound explanation can receive full credit.

## Before the next class

Read Lipman et al., [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747), abstract and
Sections 2–3. Sketch one noise–data pair, its intermediate state and its velocity target.
