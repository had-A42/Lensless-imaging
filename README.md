# Generalization of Algorithms for Lensless Cameras

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#pre-trained-checkpoints">Pre-trained Checkpoints</a> •
  <a href="#dataset">Dataset</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#citation">Citation</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

<p align="center">
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python 3.10">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-2.2-ee4c2c.svg" alt="PyTorch 2.2">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT license">
  </a>
</p>

## About

This repository contains the implementation of the research project
**Generalization of Algorithms for Lensless Cameras**. The project studies
direct, PSF-free image reconstruction: at inference time, the neural network
receives a lensless sensor measurement without the point spread function (PSF)
or mask parameters.

The main experiments evaluate reconstruction on masks that were not used for
training. They cover mask diversity, external pretraining, structured scenes,
and the gap between matched synthetic and real DigiCam measurements. The
conclusions apply to a fixed camera geometry and one family of masks.

The codebase includes:

- synthetic DigiCam measurements generated from MIRFLICKR, MNIST, and CelebA;
- real measurements from the DigiCam-Mirflickr-MultiMask datasets;
- PSF-free DRUNet and X-Restormer reconstructors;
- SD-VAE, DINOv2, and LingBot-Vision pretraining experiments;
- a PSF-aware reference implementation;
- deterministic scene splits, mask namespaces, PSF caching, and training
  resume;
- PSNR, SSIM, LPIPS, and structured-scene metrics.

## Installation

The project was developed with Python 3.10 and PyTorch 2.2. A CUDA-capable GPU
is recommended for full-resolution training. CPU and Apple MPS are suitable for
data preparation and short checks.

0. Create and activate a virtual environment:

   ```bash
   python3.10 -m venv .venv
   source .venv/bin/activate
   ```

1. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

2. Optionally install the Git hooks:

   ```bash
   pre-commit install
   ```

Keep Hugging Face downloads inside the repository:

```bash
export HF_HOME="$PWD/data/huggingface"
```

Datasets, checkpoints, generated PSFs, Hydra outputs, and W&B files are stored
in ignored directories.

## Pre-trained checkpoints

### X-Restormer

The X-Restormer experiments can start from the official GoPro deblurring
checkpoint:

```bash
mkdir -p model_weights/xrestormer
gdown 1akQHXC-PFHgj9iayHxFOm_leSc9t4RBC +  -O model_weights/xrestormer/net_g_latest.pth
export XRESTORMER_CHECKPOINT="$PWD/model_weights/xrestormer/net_g_latest.pth"
```

Expected SHA-256:

```text
12c6f5c8053cdede1e37d442ebf0824817007c32214c1b51b29e50636db39354
```

### SD-VAE and visual encoders

The SD-VAE, DINOv2, and LingBot-Vision configs accept local Hugging Face
snapshots:

```bash
hf download stabilityai/sd-vae-ft-mse +  --revision 31f26fdeee1355a5c34592e401dd41e45d25a493 +  --local-dir data/pretrained/sd-vae-ft-mse
export SD_VAE_SNAPSHOT="$PWD/data/pretrained/sd-vae-ft-mse"

hf download facebook/dinov2-small +  --local-dir data/pretrained/dinov2-small
export DINOV2_SNAPSHOT="$PWD/data/pretrained/dinov2-small"

hf download robbyant/lingbot-vision-vit-small +  --local-dir data/pretrained/lingbot-vision-vit-small
export LINGBOT_SNAPSHOT="$PWD/data/pretrained/lingbot-vision-vit-small"
```

Record the resolved snapshot revision together with each experiment. The SD-VAE download command pins the exact Hugging Face revision used in the experiments.

## Dataset

Three local scene datasets are prepared through matching top-level commands:

```bash
python prepare_mirflickr.py
python prepare_mnist.py
python prepare_celeba.py
```

Each command downloads missing source files, checks their integrity, prepares
the required images, and creates a deterministic split manifest under
`manifests/`. Manifest JSON files are local generated artifacts and are not
committed to Git.

Preparation is idempotent. An existing manifest is accepted only when it
matches the newly generated split. To replace it intentionally, use:

```bash
python prepare_mnist.py manifest.rewrite=true
```

Use `download=false` to prohibit network access and verify an existing local
copy:

```bash
python prepare_mirflickr.py download=false
```

### MIRFLICKR-25000

Synthetic natural-scene experiments use version 1 of the
[MIRFLICKR-25000 Kaggle mirror](https://www.kaggle.com/datasets/skfrost19/mirflickr25k).
The archive and extracted images require about 6.2 GiB together.

`prepare_mirflickr.py` downloads the archive, verifies its size and SHA-256,
extracts it safely, and decode-checks all 25,000 JPEG files.

```text
archive: data/raw/mirflickr25k/mirflickr25k-kaggle-v1.zip
size:    3072872067 bytes
SHA256:  f1dadfc89966c43a65d361839e57b18ab6cc9397b1919fa57afdf0590f265588
```

The default split contains 4,096 training scenes, 128 validation scenes, and
256 reserved test scenes. Images used by the external real-data test protocol
are excluded.

### MNIST

`prepare_mnist.py` downloads both official MNIST partitions. It creates a
deterministic, stratified 55,000/5,000 train-validation split from the official
training partition. The official 10,000-image test partition remains separate
and is not used by the training configs.

### CelebA

`prepare_celeba.py` downloads the official partition metadata and aligned
image archive. It extracts the deterministic set of 4,096 training and 128
validation images used by the project.

To use a different directory:

```bash
export CELEBA_ROOT=/absolute/path/to/celeba
python prepare_celeba.py
```

### Real DigiCam measurements

The full real-data configs use
[`bezzam/DigiCam-Mirflickr-MultiMask-25K`](https://huggingface.co/datasets/bezzam/DigiCam-Mirflickr-MultiMask-25K)
at the pinned revision:

```text
21d82b67662ed1e590a40c98688c32cb3c74f079
```

The dataset is downloaded automatically by `datasets.load_dataset` when a
real-data config is used. For the smaller local pilot, download the pinned 1K
snapshot explicitly:

```bash
hf download bezzam/DigiCam-Mirflickr-MultiMask-1K +  --repo-type dataset +  --revision 782d4557516d82c2797e56dde70973c25db70d69 +  --local-dir data/hf/DigiCam-Mirflickr-MultiMask-1K
```

### Synthetic masks and PSFs

Synthetic masks are reconstructed from a base seed, partition, and mask index. Training, validation, test, and streaming masks
use disjoint deterministic namespaces. Generated PSFs may be cached under
`data/psf_cache/`. The cache changes runtime, not the generated measurements.

## How To Use

All experiments use Hydra configs from `src/configs/`.

### Training

Run one experiment with:

```bash
python train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Inspect the fully resolved config before a long run:

```bash
python train.py -cn=e01_mask_diversity_screen +  scale_condition=finite_100 +  --cfg job --resolve
```

The default W&B mode is offline. Enable online logging explicitly:

```bash
wandb login
python train.py -cn=e01_mask_diversity_screen writer.mode=online
```

Run the mask-diversity screen:

```bash
python train.py -m -cn=e01_mask_diversity_screen +  scale_condition=finite_100,infinite +  trainer.seed=42,52,62 +  trainer.device=cuda
```

Run one GoPro-initialized X-Restormer experiment:

```bash
python train.py -cn=pt_xrestormer_screen +  initialization=xrestormer_gopro +  scale_condition=finite_100 +  trainer.seed=42 +  trainer.device=cuda +  trainer.amp.enabled=true +  trainer.amp.dtype=bfloat16
```

Structured-scene experiments use:

```bash
python train.py -cn=mnist_psf_free trainer.seed=42 trainer.device=cuda

python train.py -cn=celeba_psf_free +  scale_condition=finite_100 +  trainer.seed=42 +  trainer.device=cuda
```

The real-measurement baseline is launched with:

```bash
python train.py -cn=psf_free_real_train +  trainer.device=cuda +  writer.run_name=real-25k-unet8m-seed0
```

### Evaluation

Evaluate a PSF-free checkpoint on the fixed MIRFLICKR validation grid:

```bash
python inference.py -cn=e01_psf_free_eval +  condition_name=finite-100 +  inferencer.device=cuda +  inferencer.from_pretrained=/absolute/path/to/model_best.pth +  writer.run_name=e01-eval-finite-100
```

Evaluate the published PSF-aware reference:

```bash
python inference.py -cn=ref_psf_aware_real +  inferencer.device=cuda +  writer.run_name=ref-psf-aware-real
```

The reference receives the true PSF and is not a PSF-free baseline.

### Short smoke test

The following command runs one optimizer step and one validation example on
CPU. MIRFLICKR must already be prepared.

```bash
python train.py -cn=e01_mask_diversity_screen +  scale_condition=finite_100 +  trainer.device=cpu +  trainer.n_epochs=1 +  trainer.epoch_len=1 +  trainer.total_steps=1 +  trainer.monitor=off +  trainer.override=true +  dataloader_builder.validation_mask_count=1 +  dataloader_builder.validation_scenes_per_mask=1 +  dataloader_builder.num_workers=0 +  dataloader_builder.psf_cache.mode=off +  dataloader_builder.psf_cache.warmup=false +  writer.mode=disabled +  writer.run_name=readme-smoke
```

The first LPIPS run may download torchvision VGG weights.

### Outputs and reproducibility

- checkpoints, resolved configs, and logs are written to `saved/<run_name>/`;
- Hydra working directories are written below `outputs/` or `multirun/`;
- offline W&B runs are written below `wandb/`;
- validation metrics are saved both per image and per mask;
- reusing a run name requires `trainer.override=true`;
- training can be resumed with
  `trainer.resume_from=checkpoint-epochN.pth`.

For a reproducible result, record the Git commit, resolved config, dataset
revision, local manifest hash, and checkpoint hash.

## Citation

If you use this repository, please cite it as follows:

```bibtex
@software{khoroshilov2026lensless,
  author = {Khoroshilov, Andrey},
  title = {Lensless Imaging: Generalization of Algorithms for Lensless Cameras},
  year = {2026},
  url = {https://github.com/had-A42/Lensless-imaging}
}
```

## Credits

This repository is based on the
[PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template)
and uses components from
[LenslessPiCam](https://github.com/LCAV/LenslessPiCam). The X-Restormer
implementation follows the official
[X-Restormer repository](https://github.com/Andrew0613/X-Restormer); its license
notice is preserved in [`LICENSES/X-Restormer.txt`](LICENSES/X-Restormer.txt).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
