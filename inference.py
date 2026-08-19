import os
import warnings
from pathlib import Path

cache = Path(__file__).resolve().parent / ".cache" / "matplotlib"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache))

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from src.datasets.data_utils import get_dataloaders
from src.trainer import Inferencer
from src.utils.init_utils import set_random_seed, setup_saving_and_logging
from src.utils.io_utils import ROOT_PATH

warnings.filterwarnings("ignore", category=UserWarning)


@hydra.main(version_base=None, config_path="src/configs", config_name="inference")
def main(config):
    """
    Main script for inference. Instantiates the model, metrics, and
    dataloaders. Runs Inferencer to calculate metrics and (or)
    save predictions.

    Args:
        config (DictConfig): hydra experiment config.
    """
    set_random_seed(config.inferencer.seed)

    writer = None
    logger = None
    if config.get("writer") is not None:
        project_config = OmegaConf.to_container(config, resolve=True)
        logger = setup_saving_and_logging(config)
        writer = instantiate(config.writer, logger, project_config)

    if config.inferencer.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.inferencer.device

    # setup data_loader instances
    # batch_transforms should be put on device
    dataloaders, batch_transforms = get_dataloaders(config, device)

    # build model architecture, then print to console
    model = instantiate(config.model).to(device)
    if logger is None:
        print(model)
    else:
        logger.info(model)

    # get metrics
    metrics = instantiate(config.metrics)

    # save_path for model predictions
    if writer is None:
        save_path = ROOT_PATH / "data" / "saved" / config.inferencer.save_path
    else:
        save_path = ROOT_PATH / config.inferencer.save_dir / config.writer.run_name
    save_path.mkdir(exist_ok=True, parents=True)
    OmegaConf.save(config, save_path / "resolved_config.yaml", resolve=True)

    inferencer = Inferencer(
        model=model,
        config=config,
        device=device,
        dataloaders=dataloaders,
        batch_transforms=batch_transforms,
        save_path=save_path,
        metrics=metrics,
        skip_model_load=config.inferencer.get("skip_model_load", False),
        logger=logger,
        writer=writer,
    )

    try:
        logs = inferencer.run_inference()
    finally:
        if writer is not None:
            writer.finish()

    for part in logs.keys():
        for key, value in logs[part].items():
            full_key = part + "_" + key
            print(f"    {full_key:15s}: {value}")


if __name__ == "__main__":
    main()
