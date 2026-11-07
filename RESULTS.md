# Observed experiment results

## CIFAR-10 guidance

| w | Accuracy | FID | FID_class |
|---|---|---|---|
| 0 | 0.110 | 30.97 | -- |
| 1 | 0.958 | 30.93 | 111.40 |
| 1.5 | 0.986 | 30.96 | 107.16 |
| 2 | 0.995 | 32.22 | 107.87 |
| 3 | 0.991 | 35.59 | 112.81 |
| 4 | 0.990 | 38.22 | 116.79 |
## CIFAR subset 100: 20000 updates

| Update | FID |
|---|---|
| 4000 | 73.04 |
| 8000 | 88.55 |
| 12000 | 98.54 |
| 16000 | 105.63 |
| 20000 | 106.83 |

Median nearest-training-image distance: generated 1.19; real test 23.67.

## CIFAR subset 1000: 20000 updates

| Update | FID |
|---|---|
| 4000 | 92.00 |
| 8000 | 111.34 |
| 12000 | 121.21 |
| 16000 | 129.61 |
| 20000 | 132.47 |

Median nearest-training-image distance: generated 13.86; real test 21.25.

## MNIST guidance

| w | Accuracy | Classifier-feature FID |
|---|---|---|
| 0 | 0.104 | 106.40 |
| 1 | 0.964 | 22.10 |
| 2 | 1.000 | 85.14 |
| 4 | 1.000 | 182.57 |
