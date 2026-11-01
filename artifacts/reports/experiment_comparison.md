# Measured POC comparison

Selection uses validation macro-F1; official test was evaluated after freezing.

| Model | Run | Val macro-F1 | Test Top-1 | Test macro-F1 | CPU P95 ms |
|---|---|---:|---:|---:|---:|
| hog_svm | poc-v1-cpu-budget-v2-primal | 0.1949 | 0.1876 | 0.1870 | 5.76 |
| resnet18 | poc10-rtx3050-v1 | 0.8595 | 0.8976 | 0.8975 | 29.85 |
| efficientnet_b0 | poc10-rtx3050-v1 | 0.8762 | 0.8980 | 0.8970 | 20.33 |

Selected artifact: `efficientnet_b0/poc10-rtx3050-v1`.

POC 10 classes only. Scores are not calibrated probabilities. Review per-class failures and convergence/training budget before presenting results.
