# Model evaluation report

## Current POC/GPU workflow (supersedes the old CPU budget below)

POC manifest contract audit passed for 10 classes and 6000/1500/2500 images.
Current CPU HOG fit is still incomplete at the latest inspection; no real CNN release
or final test metrics exist. RTX 3050 Ti 4GB profiles (2 head + up to 8 fine-tune epochs,
batch 4/accumulation 4/AMP) and a separate CUDA environment are prepared, not executed.
Freeze all three completed candidates before any official test evaluation. The common
evaluation target is CPU, 2500 test images, batch 1, three validation warmups. Selection
will generate experiment_comparison.csv/.md from real metrics and preserve the same
manifest, label mapping and frozen protocol across methods. See POC_REAL_AI.md.
No accuracy/F1, winning method or successful real checkpoint integration is claimed yet.

Current verified state at this checkpoint: synthetic code smoke tests completed for
HOG+SVM, ResNet18 and EfficientNet-B0. These are not food dataset metrics.
Official ImageNet initialization weights for the two CNNs were downloaded with hash
checks; they are not themselves food recognition checkpoints.

The planned actual POC uses the ten fixed classes, original Food-101 test files and
stratified validation from original train. A short CPU budget is declared in
configs/experiment_budget.yaml before test evaluation: 1 head epoch + 1 fine-tune
epoch, batch size 4, seed 42 for both CNNs. This is a resource-limited experiment,
not a converged/full-101 benchmark. Baseline C is selected only on validation.

No formal Top-1, macro-F1 or winning model is entered here until scripts produce
actual metrics. Later actual results must include sample counts, manifest digest,
dataset version, hardware and latency scope. Reports should discuss the small
training budget rather than implying a mature production classifier.

Failure exports retain UNREVIEWED until someone inspects the images. Mixed dish,
plating, lighting, visual similarity and domain shift must not be guessed from class
names alone. No local Vietnamese domain-shift photos have been collected yet.
