import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm

from src.digicam_synth.pipeline import synthesize_measurement


def load_rgb(filepath: str) -> np.ndarray:
    return np.array(Image.open(filepath).convert("RGB"), dtype=np.uint8)


def collect_images(input_dir: str, exts: list[str]) -> list[Path]:
    files = []
    for ext in exts:
        files.extend(sorted(Path(input_dir).glob(f"*.{ext}")))
    return sorted(files)


@hydra.main(version_base=None, config_path="src/configs", config_name="generate_synth")
def main(config):
    print(OmegaConf.to_yaml(config))

    image_files = collect_images(config.input_dir, config.extensions)
    if config.generate_samples and len(image_files) == 0:
        raise RuntimeError(f"No images found in {config.input_dir}")

    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    (save_dir / "dataset_metadata.json").write_text(
        json.dumps(OmegaConf.to_container(config, resolve=True), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    psf_dir = save_dir / "psf"
    psf_dir.mkdir(parents=True, exist_ok=True)

    generation_summary = []

    for psf_id in tqdm(range(config.n_psf), "Mask generating"):
        seed_mask = int(config.seed_base + psf_id)
        rng = np.random.default_rng(seed_mask)

        image_files_subset = []
        if config.generate_samples:
            subset_size = min(config.samples_per_mask, len(image_files))
            idx = rng.choice(len(image_files), size=subset_size, replace=False)
            image_files_subset = [image_files[i] for i in idx]

        saved_samples = synthesize_measurement(
            config=config,
            rng=rng,
            save_dir=save_dir,
            psf_dir=psf_dir,
            psf_id=psf_id,
            image_fps=image_files_subset,
        )

        generation_summary.append(
            {
                "psf_id": psf_id,
                "seed": seed_mask,
                "num_samples": len(saved_samples),
                "sample_names": [sample["sample_name"] for sample in saved_samples],
                "psf_file": f"psf/psf_{psf_id}.png",
                "psf_metadata_file": f"psf/psf_{psf_id}.metadata.json",
            }
        )

    (save_dir / "generation_summary.json").write_text(
        json.dumps(generation_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    total_samples = sum(item["num_samples"] for item in generation_summary)
    print(f"Done! Generated {total_samples} samples in {save_dir}")


if __name__ == "__main__":
    main()