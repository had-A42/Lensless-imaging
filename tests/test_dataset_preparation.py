import hashlib
import zipfile

import pytest

from src.datasets.preparation import download_http, extract_zip, verify_file


def test_local_download_and_multi_digest_verification(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"lensless-data")
    destination = tmp_path / "downloads" / "dataset.bin"

    assert download_http(source.as_uri(), destination, enabled=True)
    assert destination.read_bytes() == source.read_bytes()
    assert not download_http(source.as_uri(), destination, enabled=False)

    observed = verify_file(
        destination,
        expected_bytes=len(b"lensless-data"),
        expected_digests={
            "md5": hashlib.md5(b"lensless-data").hexdigest(),
            "sha256": hashlib.sha256(b"lensless-data").hexdigest(),
        },
    )
    assert set(observed["digests"]) == {"md5", "sha256"}


def test_download_is_not_published_before_integrity_check(tmp_path):
    source = tmp_path / "source.bin"
    source.write_bytes(b"corrupted")
    destination = tmp_path / "downloads" / "dataset.bin"

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        download_http(
            source.as_uri(),
            destination,
            enabled=True,
            expected_digests={"sha256": hashlib.sha256(b"expected").hexdigest()},
        )

    assert not destination.exists()
    assert not destination.with_suffix(".bin.part").exists()


def test_extract_zip_supports_selected_members(tmp_path):
    archive = tmp_path / "dataset.zip"
    with zipfile.ZipFile(archive, "w") as file:
        file.writestr("images/one.txt", "one")
        file.writestr("images/two.txt", "two")

    output = tmp_path / "output"
    assert extract_zip(archive, output, members=["images/two.txt"]) == 1
    assert not (output / "images" / "one.txt").exists()
    assert (output / "images" / "two.txt").read_text() == "two"


def test_extract_zip_rejects_parent_traversal(tmp_path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as file:
        file.writestr("../outside.txt", "unsafe")

    with pytest.raises(ValueError, match="Unsafe zip member"):
        extract_zip(archive, tmp_path / "output")
    assert not (tmp_path / "outside.txt").exists()
