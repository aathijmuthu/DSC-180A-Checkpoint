# Experiment notes

## Predictions recorded before running experiments

- At high CIFAR noise levels, denoiser estimates should show broad structure or class averages. Fine details should emerge at lower noise. The exact transition must be read from the trajectories.
- With starting noise fixed, changing the requested class should change semantic content while potentially retaining some spatial or color structure.
- Increasing guidance should initially improve requested-class accuracy, but may reduce variation or introduce artifacts. We do not assume a monotonic improvement in accuracy or FID.
- Models trained on 100 or 1,000 CIFAR images may memorize. Compare nearest-training-image distances of generated images with those of real test images, and inspect pairs visually.
- MNIST w=0 should ignore the request. Larger w may improve target matching at the expense of within-class variation. The evaluator can confidently misread malformed outputs.

## Required observations after GPU runs

- CIFAR trajectories (cat, seed 2): at what sigma is the class first visible for w=1 and w=4? From which sigma does each image barely change? How do the estimates differ at low noise?
- Which CIFAR guidance weight would you choose, using accuracy, FID, FID_class, and visual inspection? Record ambiguities and artifacts.
- For each training subset: describe FID versus update and sample/test nearest-neighbor distances. Pixel proximity is evidence of similarity, not proof of exact copying.
- For MNIST: describe the accuracy/FID tradeoff for w=0,1,2,4 and identify malformed or ambiguous samples.

## Observations from completed local checks

- Frozen evaluator real-test accuracy: CIFAR-10 94.4% on all 10,000 test images; MNIST 99.32% on all 10,000 test digits. These evaluate classifiers, not generated samples.
- The unconditional CIFAR trajectory (seed 0, three images) starts with nearly uniform blurred denoiser estimates at sigma=80. Recognizable content appears between the displayed sigma=3.3 and sigma=1.1 panels, depending on the image. Composition is broadly stable by sigma=0.3, with later sharpening. These three examples do not establish a general decision threshold.
- Conditional trajectory answers (cat, seed 2): cat structure is evident by sigma=1.1 for both w=1 and w=4. In the second example, w=4 already shows clearer ears/face at sigma=3.3 than w=1. This supports earlier class visibility in that example, not a universal threshold. Composition changes little by sigma=0.3; later steps sharpen detail. At low noise, w=4 increases contrast and pronounced fur/face features and can change pose or proportions.
- Cat guidance grid: w=0 produces diverse object classes, whereas conditional rows move toward cats. Larger weights sharpen several faces, but the sixth column stays dark and ambiguous and the first column changes body shape. Larger w is not uniformly preferable by visual inspection alone. A preferred weight awaits the numerical sweep.
- MNIST evaluation at 6,500 updates, 1,000 images per weight: accuracy is 10.2%, 95.7%, 99.9%, 100% for w=0,1,2,4. Classifier-feature FID is 135.30, 24.21, 93.99, 210.66. Among tested weights, w=1 best matches the real feature distribution, while w=2 improves class accuracy at a substantial distribution-distance cost. In the paired first-ten grid, the requested seven is ambiguous at w=1 and clearer at larger weights. Within-class pixel spread decreases with guidance, but this alone is not proof of better images or generalization.
- The laptop was closed during local training. Elapsed-time estimates included the suspended interval and overstated active compute time. After resuming from update 6,500, a fresh 20-update benchmark measured approximately 0.16 seconds/update. Keep the lid open; an idle-sleep inhibitor does not guarantee operation with the lid closed.
- Final MNIST run: 10,000 updates, 1,000 images per weight, all four evaluations complete. At w=0,1,2,4, accuracy is 10.4%, 96.4%, 100%, 100%; classifier-feature FID is 106.40, 22.10, 85.14, 182.57. The preferred measured distribution match is w=1. If near-perfect requested-digit matching is the priority, w=2 achieves the same measured accuracy as w=4 with substantially lower feature FID. Accuracy is an evaluator's decision, not proof of image quality.
- MNIST reproducibility detail: the local run resumed from update 6,500. That early checkpoint predated MPS RNG persistence, so CPU batch selection was restored but the MPS training-noise stream restarted from seed 0. Current checkpoints save CPU, CUDA, and MPS RNG states. All reported guidance settings use the same final checkpoint and paired evaluation noise; the interruption does not change that controlled comparison.
- CIFAR training was checked with the exact 7M-parameter EDM architecture and EDMLoss on CPU and MPS, including backward and optimizer steps. A 20-update batch-64 MPS benchmark measured about 0.52 seconds/update while reference-feature extraction was running. The real 100-image training run is active; the initial stage ends at update 4,000 and the queued continuation keeps the 20,000-update snapshot schedule.
- The optional full-data EDM dataset export was verified: 50,000 PNGs, 50,000 metadata labels, exactly 5,000 examples per class, and a valid ZIP archive. The optional full-network training run has not been launched.
- DSMLP SSH was connected with the user's interactive authentication. A one-GPU batch job is queued as `dsc180-checkpoint`; account policy limits it to the small-GPU pool, which was busy at submission. The 1,500-update CIFAR subset checkpoint was transferred. Course containers mount the personal project at `/home/amuthu/private/DSC-180A-Checkpoint`.
- Inception optimization: on four identical synthetic uint8 inputs, CPU/MPS features differed by at most 4.95e-6 (relative RMS 1.12e-6). A warmed 32-image MPS batch took about 0.84 seconds. The local sweep was restarted with MPS extraction while preserving all 50,000 real reference images. Float64 conversion and covariance calculations run on CPU. Earlier incomplete CPU reference extraction produced no reported FID results.
- A Colab notebook was prepared as a GPU fallback. It uses the same required CIFAR settings and exports numerical results and figures for local review; its execution requires an available user-selected GPU runtime.
- Local numerical checks passed for FID identity/translation/symmetry, null-guidance independence from the request, null-label embedding gradients under dropout, the constant-denoiser sampler, checkpoint/optimizer/RNG restoration, and figure saving. A two-update MNIST run and a one-update resume also completed. Those generated outputs are smoke checks, not reported assignment results.

## Protocol decisions

- CIFAR evaluation: 1,000 generated images, balanced requested classes, 18 Heun steps; all-class FID uses NVIDIA's training reference and FID_class uses real training images of each requested class. FID_class at w=0 is omitted because requests are ignored.
- MNIST evaluation: 1,000 generated images, balanced requested digits, 32 deterministic DDIM steps, seed 1000; reference is all 10,000 held-out test digits, features are the frozen lab classifier's 128-dimensional representation. Report this as classifier-feature FID, not Inception FID.
- MNIST training: all 60,000 training digits, fresh U-Net initialization (ch=64), uniform time on the lab cosine path, clean-image MSE, 10% label dropout, Adam at 2e-4, EMA. The assignment does not prescribe an update count; default budget is 10,000 updates of 128 images. This objective is our implementation choice; the supplied MNIST notebook contains no training loop.
- Small CIFAR training: the notebook's 7M-parameter SongUNet/EDM preconditioner, EDMLoss, Adam at 2e-4, EMA, 10% label dropout, 20,000 updates of 64 images, five evaluated snapshots.
- The 50,000-image full EDM training run is optional and has not been launched.
- Any shortened training run must report its actual updates; no smoke run is a checkpoint result.
