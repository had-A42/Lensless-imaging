DEFAULT_MASK_SEED = 42
MASK_COUNTS = {"train": 10_000, "validation": 32, "test": 100}
NAMESPACES = {"train": 1, "validation": 2, "test": 3, "infinite": 4}


def mask_seed(base_seed, partition, index):
    if partition not in NAMESPACES:
        raise ValueError(f"unknown mask partition: {partition}")
    if not 0 <= base_seed < 2**31:
        raise ValueError("base_seed must fit in 31 bits")
    if not 0 <= index < 2**24:
        raise ValueError("mask index must fit in 24 bits")
    return (int(base_seed) << 32) | (NAMESPACES[partition] << 24) | int(index)


def get_mask_records(base_seed, partition, count=None, allow_test=False):
    if partition not in MASK_COUNTS:
        raise ValueError(f"unknown mask partition: {partition}")
    if partition == "test" and not allow_test:
        raise ValueError("test masks are closed during training and validation")
    if count is None:
        count = MASK_COUNTS[partition]
    count = int(count)
    if not 0 < count <= MASK_COUNTS[partition]:
        raise ValueError(f"invalid {partition} mask count: {count}")
    return [
        {
            "mask_id": f"{partition}_{index:05d}",
            "mask_seed": mask_seed(base_seed, partition, index),
        }
        for index in range(count)
    ]
