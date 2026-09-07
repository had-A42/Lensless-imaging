import os
import shutil
import tempfile
import zipfile
from pathlib import Path

cache = Path(__file__).resolve().parent / ".cache" / "matplotlib"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache))

import hydra
from hydra.utils import to_absolute_path
from torchvision.datasets import CelebA
from tqdm import tqdm
import gdown

from src.datasets.celeba import celeba_filenames


@hydra.main(version_base=None, config_path="src/configs", config_name="celeba_prepare")
def main(config):
    root = Path(to_absolute_path(config.root_dir))
    root.mkdir(parents=True, exist_ok=True)
    file_ids = {name: file_id for file_id, _, name in CelebA.file_list}

    def get_file(name):
        path = root / name
        if path.is_file():
            return path
        if not config.download:
            raise FileNotFoundError(
                f"Place {name} in {root}, or run prepare_celeba.py download=true."
            )
        with tempfile.TemporaryDirectory(
            prefix="celeba-download-", dir=root
        ) as directory:

            temporary = Path(directory) / name
            gdown.download(id=file_ids[name], output=str(temporary), quiet=False)
            if name.endswith(".zip") and not zipfile.is_zipfile(temporary):
                raise ValueError("Google Drive did not return the CelebA ZIP archive")
            if name.endswith(".txt"):
                with temporary.open() as file:
                    first_row = file.readline().split()
                if (
                    len(first_row) != 2
                    or not first_row[0].endswith(".jpg")
                    or first_row[1] != "0"
                ):
                    raise ValueError("Google Drive did not return the CelebA partition")
            temporary.replace(path)
        return path

    get_file("list_eval_partition.txt")
    names = []
    for split, count in config.samples.items():
        names.extend(celeba_filenames(root, split, int(count), int(config.split_seed)))
    missing = [
        name for name in names if not (root / "img_align_celeba" / name).is_file()
    ]
    if missing:
        archive = get_file("img_align_celeba.zip")
        (root / "img_align_celeba").mkdir(exist_ok=True)
        with zipfile.ZipFile(archive) as file:
            for name in tqdm(missing, desc="Extracting train/validation faces"):
                destination = root / "img_align_celeba" / name
                temporary = destination.with_suffix(".jpg.part")
                with file.open(f"img_align_celeba/{name}") as source, temporary.open(
                    "wb"
                ) as output:
                    shutil.copyfileobj(source, output)
                temporary.replace(destination)
    print(
        f"Ready: {dict(config.samples)}, subset seed {config.split_seed}, root {root}"
    )


if __name__ == "__main__":
    main()
