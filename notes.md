# Experiment notes

## Predictions recorded before the checkpoint experiments

- At high CIFAR noise levels, denoiser estimates should show broad structure or class averages. Fine details should emerge at lower noise. The exact transition must be read from the trajectories.
- With starting noise fixed, changing the requested class should change semantic content while potentially retaining some spatial or color structure.
- Increasing guidance should initially improve requested-class accuracy, but may reduce variation or introduce artifacts. We do not assume a monotonic improvement in accuracy or FID.
- Models trained on 100 or 1,000 CIFAR images may memorize. Compare nearest-training-image distances of generated images with those of real test images, and inspect pairs visually.
- MNIST w=0 should ignore the request. Larger w may improve target matching at the expense of within-class variation. The evaluator can confidently misread malformed outputs.

## CIFAR notebook answers

1. **High noise and unconditional trajectory:** At sigma=80, the clean-image estimate should be a smooth image near the data average. In the three saved seed-0 examples, recognizable content emerges between sigma=3.3 and 1.1. Composition changes little below about 0.3, while detail continues to sharpen. These thresholds describe these examples.
2. **Changing the class with paired noise:** Coarse position, colors, and background can persist down a column, while object identity changes. The condition does not prescribe every pixel.
3. **Increasing guidance:** Stronger guidance should emphasize the requested class but can reduce variation or distort shapes. The cat grid sharpens several faces, yet its sixth column remains dark and ambiguous at w=4.
4. **Conditional trajectory questions (cat, seed 2):** Cats are recognizable by sigma=1.1 for both w=1 and w=4. The second w=4 image already has clearer ears/face at sigma=3.3. Composition is broadly stable below sigma=0.3; at low noise, w=4 strengthens contrast and face/fur detail and changes some poses or proportions.
5. **2D FID exercise:** The shifted-mean panel has the largest FID, 2.27, almost entirely from its mean term. The low-variety panel has FID 0.69, mainly from covariance. Even the same-distribution, 30-point panel has FID about 0.01 because of finite-sample estimation.
6. **Guidance choice and metric limits:** I favor w=1.5: 98.6% accuracy, global FID 30.96, and the lowest class FID, 107.16. Weight two gives the highest accuracy, 99.5%; larger weights worsen FID without improving accuracy. Accuracy can miss malformed images and lost diversity. Global FID ignores requests; class FID adds class information but has only 100 generated images per class. Visual checks remain necessary, and one paired seed does not establish significance for small score differences.

The [completed CIFAR notebook](dsc180-d32-cifar10-edm-lab/cifar10_edm_lab.ipynb) contains the runnable trajectory exercise, written answers, saved-figure links, and full sweep table. The sweep uses 1,000 balanced samples per weight and all 50,000 real training images for class references.

## Restricted-data CIFAR observations

Both models completed 20,000 updates and five 1,000-image unconditional snapshot evaluations.

| Training images | FID at 4k / 8k / 12k / 16k / 20k | First/last snapshot mean loss | Generated/test nearest-image medians |
|---|---|---|---|
| 100 | 73.04 / 88.55 / 98.54 / 105.63 / 106.83 | 0.0367 / 0.0179 | 1.19 / 23.67 |
| 1,000 | 92.00 / 111.34 / 121.21 / 129.61 / 132.47 | 0.1788 / 0.0694 | 13.86 / 21.25 |

Loss means use the 200 updates before each endpoint snapshot. Nearest-image comparisons use 100 generated and 100 real test images. The 100-image model closely reproduces training objects and backgrounds, supporting memorization. The 1,000-image model has blurry partial matches; smaller pixel distance alone can also reflect averaging. These fixed-budget runs do not show that more data harms diffusion models.

At w=0,1,2,4, final CFG-grid accuracies over 100 samples per weight are 7%,96%,100%,100% for 100 training images and 10%,47%,74%,84% for 1,000 training images. Guidance improves matching, but several larger-subset outputs remain ambiguous. These grid scores are separate from the pretrained 1,000-image sweep.

## MNIST checkpoint observations

The new U-Net trained from scratch on all 60,000 digits for 10,000 updates. At w=0,1,2,4, accuracy is 10.4%,96.4%,100%,100%; classifier-feature FID is 106.40,22.10,85.14,182.57. Each setting uses 1,000 balanced samples against all 10,000 test digits in the frozen classifier's 128-dimensional feature space.

I favor w=1 for the closest measured feature distribution, or w=2 for class matching. Weight four adds no measured accuracy gain and worsens FID. The paired requested seven is clearer at higher weights; classifier agreement alone does not prove image quality.

## MNIST reference-lab exercises

Prediction recorded before the final-audit run: changing the request should change digit identity without fixing every pixel; stronger guidance may improve matching but can reduce useful variation or exaggerate artifacts. The new comparison fixes seed 19, 40 images, and 16 steps.

The [completed notebook](dsc180-d32-mnist-guidance-lab/guidance_mnist.ipynb) includes executed Parts 1–4, four grids, metric outputs, the 199-word comparison, and architecture answers. With the supplied frozen generator, requests 3/8 reach 100%/95% match at w=1. Target-three match at w=0,1,2,5 is 5%,100%,100%,100%; pixel spread at w=1,2,5 is 0.2118,0.1924,0.1933. The sixth displayed eight has crowded loops requiring visual review. No FID is reported for these 40-image exercises. These results are distinct from checkpoint scratch training.

## Reproducibility and optional scope

- Frozen-classifier real-test accuracies: CIFAR-10 94.4%; MNIST 99.32%, each on 10,000 real test images.
- MNIST resumed at update 6,500 after laptop sleep. CPU batch-selection state was restored, but the early checkpoint did not preserve MPS RNG state, so the training-noise stream restarted. Current checkpoints save CPU/CUDA/MPS RNG states. All guidance comparisons use the same final weights and paired evaluation noise.
- The 100-image CIFAR run resumed a 1,500-update Mac checkpoint on DSMLP CUDA. The 1,000-image run used CUDA from scratch. Final CIFAR figures and metrics use DSMLP outputs; local pilots are excluded from reported results.
- The encouraged full-size EDM run was prepared but not run: the 50,000-image PNG archive, labels, class counts, and ZIP integrity were verified. No full-network training result is claimed.
- E7 in `part2_mnist.ipynb`, trained-network trajectories, and the further-data/CFG extensions are optional and were not run.
