from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from venusfold.data.msa.search import (
    _safe_extract_tar,
    _write_processed_msas,
    first_a3m_query,
    search_protein_msas,
)


def test_write_processed_msas_preserves_query_and_splits_taxonomy(tmp_path: Path) -> None:
    query = "ACDE"
    raw = tmp_path / "0.a3m"
    raw.write_text(
        ">query_0\nACDE\n"
        ">uniref_hit\nAC-E\n"
        ">query_0\nACDE\n"
        ">environment_hit\nAcCDE\n"
        ">malformed_hit\nACD\n"
    )

    _write_processed_msas(query, raw, tmp_path, {"uniref_hit": "9606"})

    paired = tmp_path / "pairing.a3m"
    unpaired = tmp_path / "non_pairing.a3m"
    assert first_a3m_query(paired) == query
    assert first_a3m_query(unpaired) == query
    assert "UniRef100_uniref_hit_9606/" in paired.read_text()
    assert "environment_hit" in unpaired.read_text()
    assert "malformed_hit" not in unpaired.read_text()


def test_search_rejects_cached_raw_msa_for_another_query(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "0.a3m").write_text(">query_0\nWRONG\n")

    def reuse_existing(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "venusfold.data.msa.search._submit_and_download", reuse_existing
    )
    with pytest.raises(ValueError, match="Raw MSA query mismatch"):
        search_protein_msas(["ACDE"], tmp_path)


def test_safe_extract_tar_rejects_parent_path(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        payload = b"unsafe"
        member = tarfile.TarInfo("../outside.txt")
        member.size = len(payload)
        handle.addfile(member, io.BytesIO(payload))

    with pytest.raises(ValueError, match="Unsafe path"):
        _safe_extract_tar(archive, tmp_path / "result")


def test_safe_extract_tar_rejects_links(tmp_path: Path) -> None:
    archive = tmp_path / "link.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        member = tarfile.TarInfo("link")
        member.type = tarfile.SYMTYPE
        member.linkname = "/etc/passwd"
        handle.addfile(member)

    with pytest.raises(ValueError, match="Links are not allowed"):
        _safe_extract_tar(archive, tmp_path / "result")
