"""Aggregate complete evaluations; paired effects are computed within training seed."""

import argparse
import statistics as st
from collections import defaultdict
from pathlib import Path

from .common import (
    ARMS,
    METRICS,
    SEEDS,
    balanced_metrics,
    read_csv,
    read_json,
    save_json,
    verify_package,
    write_csv,
)
from .run import validated_evaluation


def summarize(package):
    package = Path(package).resolve()
    plan = verify_package(package)
    seeds = plan.get("training_seeds", list(SEEDS))
    expected = len(plan["evaluations"])
    values, rows, missing = {}, [], []
    expected_hashes = None
    for job in plan["evaluations"]:
        proof = validated_evaluation(package, plan, job)
        if proof is None:
            missing.append(job["id"])
            continue
        image_rows = read_csv(package / job["output"] / "validation/per_image.csv")
        fingerprints = {
            (r["mask_id"], r["scene_id"]): (r["measurement_sha256"], r["target_sha256"])
            for r in image_rows
        }
        if expected_hashes is None:
            expected_hashes = fingerprints
        elif expected_hashes != fingerprints:
            raise ValueError(
                "Compared evaluations used different measurements or targets"
            )
        for grid_name, grid_path in (
            ("primary128", "grid128.json"),
            ("legacy32", "grid32.json"),
        ):
            grid = read_json(package / grid_path)
            keys = {(r["mask_id"], r["scene_id"]) for r in grid}
            selected = [r for r in image_rows if (r["mask_id"], r["scene_id"]) in keys]
            metrics = balanced_metrics(selected, grid)
            for metric, value in metrics.items():
                key = (
                    job["training_scenes"],
                    job["arm"],
                    job["seed"],
                    grid_name,
                    metric,
                )
                if key in values:
                    raise ValueError("Duplicate training replica")
                values[key] = value
                rows.append(
                    dict(
                        training_scenes=key[0],
                        initialization=key[1],
                        seed=key[2],
                        grid=grid_name,
                        metric=metric,
                        value=value,
                    )
                )
    effects = []
    for seed in seeds:
        for grid in ("primary128", "legacy32"):
            for metric in METRICS:

                def add(name, terms):
                    if all(key in values for _, key in terms):
                        effects.append(
                            dict(
                                contrast=name,
                                seed=seed,
                                grid=grid,
                                metric=metric,
                                delta=sum(sign * values[key] for sign, key in terms),
                            )
                        )

                for arm in ARMS:
                    add(
                        f"scenes16384-minus4096/{arm}",
                        [
                            (1, (16384, arm, seed, grid, metric)),
                            (-1, (4096, arm, seed, grid, metric)),
                        ],
                    )
                for n in (4096, 16384):
                    add(
                        f"gopro-minus-scratch/scenes{n}",
                        [
                            (1, (n, "gopro", seed, grid, metric)),
                            (-1, (n, "scratch", seed, grid, metric)),
                        ],
                    )
                add(
                    "change-in-pretraining-effect",
                    [
                        (1, (16384, "gopro", seed, grid, metric)),
                        (-1, (16384, "scratch", seed, grid, metric)),
                        (-1, (4096, "gopro", seed, grid, metric)),
                        (1, (4096, "scratch", seed, grid, metric)),
                    ],
                )
    groups = defaultdict(list)
    for row in rows:
        groups[
            (
                f"scenes{row['training_scenes']}/{row['initialization']}",
                row["grid"],
                row["metric"],
            )
        ].append(row["value"])
    for row in effects:
        groups[(row["contrast"], row["grid"], row["metric"])].append(row["delta"])
    aggregate = [
        dict(
            comparison=k[0],
            grid=k[1],
            metric=k[2],
            mean=st.mean(v),
            sample_sd=st.stdev(v) if len(v) > 1 else "",
            n=len(v),
            status="complete" if len(v) == len(seeds) else "incomplete",
        )
        for k, v in sorted(groups.items())
    ]
    dest = package / "results"
    dest.mkdir(exist_ok=True)
    write_csv(
        dest / "per_seed.csv",
        rows,
        ["training_scenes", "initialization", "seed", "grid", "metric", "value"],
    )
    write_csv(
        dest / "paired_effects.csv",
        effects,
        ["contrast", "seed", "grid", "metric", "delta"],
    )
    write_csv(
        dest / "summary.csv",
        aggregate,
        ["comparison", "grid", "metric", "mean", "sample_sd", "n", "status"],
    )
    save_json(
        dest / "status.json",
        {
            "complete": not missing,
            "evaluations_complete": len(plan["evaluations"]) - len(missing),
            "evaluations_expected": expected,
            "training_seeds": seeds,
            "statistical_scope": (
                "single-seed pilot; no across-seed SD"
                if len(seeds) == 1
                else "multiseed comparison"
            ),
            "missing": missing,
            "aggregation": "image -> mask-balanced -> training seed; sample SD, not CI",
            "LPIPS_negative_delta_is_improvement": True,
        },
    )
    lines = [
        "# SS01: масштаб обучающих сцен",
        "",
        f"Готово оценок: {expected-len(missing)}/{expected}.",
        "",
        "Результаты относятся к выборке разработки. "
        + (
            "Один seed: разброс между обучениями не оценивался."
            if len(seeds) == 1
            else "Разброс вычисляется между seeds обучения."
        ),
        "",
        "| Сравнение | Сетка | Метрика | Среднее | SD | n | Статус |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for r in aggregate:
        sd = f"{r['sample_sd']:.6f}" if r["sample_sd"] != "" else "—"
        lines.append(
            f"| {r['comparison']} | {r['grid']} | {r['metric']} | {r['mean']:.6f} | {sd} | {r['n']} | {r['status']} |"
        )
    if missing:
        lines += [
            "",
            "Неполные сравнения не являются итоговыми результатами.",
            "",
            "Ожидаются: " + ", ".join(missing),
        ]
    (dest / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print(
        f"Saved summaries for {expected-len(missing)}/{expected} evaluations in {dest}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", default="outputs/scene_scaling_20260907")
    summarize(parser.parse_args().package)
