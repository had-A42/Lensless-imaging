"""Reusable download, archive, and verification helpers for dataset setup."""

from __future__ import annotations

import hashlib
import shutil
import stat
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

from PIL import Image
from tqdm import tqdm


def file_digests(
    path: str | Path,
    algorithms: Iterable[str] = ("md5", "sha256"),
) -> dict[str, str]:
    """Calculate several hashes in one pass over a file."""
    digests = {name: hashlib.new(name) for name in algorithms}
    source = Path(path)
    with source.open("rb") as stream, tqdm(
        total=source.stat().st_size,
        desc=f"Checking {source.name}",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as progress:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            for digest in digests.values():
                digest.update(chunk)
            progress.update(len(chunk))
    return {name: digest.hexdigest() for name, digest in digests.items()}


def verify_file(
    path: str | Path,
    *,
    expected_bytes: int | None = None,
    expected_digests: dict[str, str | None] | None = None,
) -> dict[str, object]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    observed_bytes = source.stat().st_size
    if expected_bytes is not None and observed_bytes != int(expected_bytes):
        raise ValueError(
            f"File size mismatch for {source}: expected {int(expected_bytes)}, "
            f"got {observed_bytes}"
        )
    expected_digests = expected_digests or {}
    requested = [
        name for name, expected in expected_digests.items() if expected is not None
    ]
    observed = file_digests(source, requested) if requested else {}
    for name, expected in expected_digests.items():
        if expected is None:
            continue
        if observed[name].lower() != str(expected).lower():
            raise ValueError(
                f"File {name.upper()} mismatch for {source}: expected {expected}, "
                f"got {observed[name]}"
            )
    return {"bytes": observed_bytes, "digests": observed}


def download_http(
    url: str,
    destination: str | Path,
    *,
    enabled: bool,
    expected_bytes: int | None = None,
    expected_digests: dict[str, str | None] | None = None,
) -> bool:
    target = Path(destination)
    if target.is_file():
        verify_file(
            target,
            expected_bytes=expected_bytes,
            expected_digests=expected_digests,
        )
        return False
    if not enabled:
        raise FileNotFoundError(
            f"Dataset file is missing: {target}. Re-run with download=true."
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.is_file() else 0
    headers = {"User-Agent": "lensless-imaging-dataset-preparation/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(str(url), headers=headers)
    with urllib.request.urlopen(request) as response:
        resumed = offset > 0 and getattr(response, "status", None) == 206
        mode = "ab" if resumed else "wb"
        initial = offset if resumed else 0
        content_length = response.headers.get("Content-Length")
        total = initial + int(content_length) if content_length is not None else None
        with partial.open(mode) as output, tqdm(
            total=total,
            initial=initial,
            desc=f"Downloading {target.name}",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as progress:
            for chunk in iter(lambda: response.read(1024 * 1024), b""):
                output.write(chunk)
                progress.update(len(chunk))
    try:
        verify_file(
            partial,
            expected_bytes=expected_bytes,
            expected_digests=expected_digests,
        )
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(target)
    return True


def _safe_zip_member(member: zipfile.ZipInfo, root: Path) -> Path:
    destination = (root / member.filename).resolve()
    if root != destination and root not in destination.parents:
        raise ValueError(f"Unsafe zip member path: {member.filename!r}")
    mode = member.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise ValueError(
            f"Symbolic links are not allowed in dataset ZIPs: {member.filename!r}"
        )
    return destination


def extract_zip(
    archive: str | Path,
    output_dir: str | Path,
    *,
    members: Iterable[str] | None = None,
) -> int:
    """Safely extract all or selected ZIP members and return the file count."""
    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        if members is None:
            selected = source.infolist()
        else:
            selected = [source.getinfo(name) for name in members]
        destinations = [(member, _safe_zip_member(member, root)) for member in selected]
        files = [(member, path) for member, path in destinations if not member.is_dir()]
        for member, destination in tqdm(files, desc="Extracting files"):
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_suffix(destination.suffix + ".part")
            with source.open(member) as input_stream, temporary.open("wb") as output:
                shutil.copyfileobj(input_stream, output)
            temporary.replace(destination)
        for member, destination in destinations:
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
    return len(files)


def verify_images(paths: Iterable[str | Path], *, description: str) -> int:
    """Decode-check image files and return their count."""
    paths = [Path(path) for path in paths]
    for path in tqdm(paths, desc=description):
        try:
            with Image.open(path) as image:
                image.verify()
        except Exception as error:
            raise ValueError(f"Invalid image: {path}") from error
    return len(paths)


__all__ = [
    "download_http",
    "extract_zip",
    "file_digests",
    "verify_file",
    "verify_images",
]
