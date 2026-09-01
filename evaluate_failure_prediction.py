import os
import warnings
from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate, to_absolute_path

from src.datasets.data_utils import get_dataloaders
from src.failure_prediction import (
    collect_failure_data,
    collect_measurement_reference,
    evaluate_failure_predictors,
    save_failure_results,
)
from src.utils.init_utils import set_random_seed

warnings.filterwarnings("ignore", category=UserWarning)

cache = Path(__file__).resolve().parent / ".cache" / "matplotlib"
cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache))


def _load_model(model_config, checkpoint_path, device):
    model = instantiate(model_config)
    checkpoint = torch.load(
        to_absolute_path(checkpoint_path),
        map_location="cpu",
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return model.to(device).eval()


@hydra.main(
    version_base=None,
    config_path="src/configs",
    config_name="failure_prediction",
)
def main(config):
    set_random_seed(config.seed)
    if config.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = config.device

    checkpoint_paths = [path for path in config.checkpoint_paths if path is not None]
    if len(checkpoint_paths) < 2:
        raise ValueError("failure prediction needs at least two ensemble checkpoints")
    models = [
        _load_model(config.model, checkpoint_path, device)
        for checkpoint_path in checkpoint_paths
    ]
    dataloaders, _ = get_dataloaders(config, device)
    reference_measurements = collect_measurement_reference(
        dataloaders["reference"],
        device,
    )
    calibration = collect_failure_data(
        models,
        dataloaders["calibration"],
        device,
        "calibration",
    )
    test = collect_failure_data(models, dataloaders["test"], device, "test")
    scores, ridge, summary, curves = evaluate_failure_predictors(
        calibration,
        test,
        reference_measurements=reference_measurements,
        regularization=config.ridge_regularization,
    )
    summary["ensemble_size"] = len(models)
    summary["checkpoints"] = [str(path) for path in checkpoint_paths]
    output_dir = to_absolute_path(config.output_dir)
    save_failure_results(
        output_dir,
        calibration,
        test,
        scores,
        ridge,
        summary,
        curves,
    )
    print(f"Saved failure-prediction results to {output_dir}")
    for name, values in summary["predictors"].items():
        print(
            f"{name}: spearman={values['spearman']:.3f}, "
            f"AUROC={values['worst_quartile_auroc']:.3f}, "
            f"AURC={values['aurc_mse']:.6f}"
        )


if __name__ == "__main__":
    main()
