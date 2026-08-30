import json
from collections import OrderedDict
from pathlib import Path

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def read_json(fname):
    """
    Read the given json file.

    Args:
        fname (str): filename of the json file.
    Returns:
        json (list[OrderedDict] | OrderedDict): loaded json.
    """
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)
