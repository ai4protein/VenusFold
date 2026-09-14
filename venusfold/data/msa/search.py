"""Protein MSA search through an MMseqs2-compatible web service."""

from __future__ import annotations

import logging
import os
import tarfile
import time
from pathlib import Path
from typing import Mapping, Sequence

import requests
from requests.auth import HTTPBasicAuth


logger = logging.getLogger(__name__)
DEFAULT_MSA_SERVER_URL = "https://protenix-server.com/api/msa"


def _request_with_retries(
    method: str,
    url: str,
    *,
    attempts: int = 6,
    **kwargs,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with requests.Session() as session:
                session.trust_env = os.environ.get(
                    "VENUSFOLD_MSA_USE_ENV_PROXY", "false"
                ).lower() in {"1", "true", "yes"}
                response = session.request(method, url, timeout=(6.02, 120), **kwargs)
                response.raise_for_status()
                return response
        except (requests.RequestException, OSError) as error:
            last_error = error
            if attempt == attempts:
                break
            delay = min(5 * attempt, 30)
            logger.warning(
                "MSA request failed (%d/%d): %s; retrying in %ds",
                attempt,
                attempts,
                error,
                delay,
            )
            time.sleep(delay)
    raise RuntimeError(f"MSA request failed after {attempts} attempts: {url}") from last_error


def _safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    output_root = output_dir.resolve()
    with tarfile.open(archive_path) as archive:
        for member in archive.getmembers():
            if member.issym() or member.islnk():
                raise ValueError(f"Links are not allowed in MSA archive: {member.name}")
            member_path = (output_dir / member.name).resolve()
            if output_root not in member_path.parents and member_path != output_root:
                raise ValueError(f"Unsafe path in MSA archive: {member.name}")
        archive.extractall(output_dir)


def _query_fasta(sequences: Sequence[str]) -> str:
    return "".join(
        f">query_{index}\n{sequence}\n" for index, sequence in enumerate(sequences)
    ).rstrip("\n")


def _submit_and_download(
    sequences: Sequence[str],
    output_dir: Path,
    *,
    host_url: str,
    email: str,
    poll_interval: int,
    max_wait_seconds: int,
) -> None:
    archive_path = output_dir / "out.tar.gz"
    if archive_path.is_file():
        logger.info("Reusing cached MSA archive: %s", archive_path)
        _safe_extract_tar(archive_path, output_dir)
        return

    started_at = time.monotonic()

    def check_deadline() -> None:
        if time.monotonic() - started_at >= max_wait_seconds:
            raise TimeoutError(
                f"MSA search did not complete within {max_wait_seconds}s"
            )

    headers = {"User-Agent": "venusfold/1.0"}
    auth = HTTPBasicAuth("example_user", "example_password")
    response = _request_with_retries(
        "POST",
        f"{host_url.rstrip('/')}/ticket/msa",
        data={"q": _query_fasta(sequences), "mode": "env", "email": email},
        headers=headers,
        auth=auth,
    )
    result = response.json()
    while result.get("status") in {"UNKNOWN", "RATELIMIT"}:
        check_deadline()
        logger.warning("MSA server status %s; resubmitting", result.get("status"))
        time.sleep(max(poll_interval, 60))
        response = _request_with_retries(
            "POST",
            f"{host_url.rstrip('/')}/ticket/msa",
            data={"q": _query_fasta(sequences), "mode": "env", "email": email},
            headers=headers,
            auth=auth,
        )
        result = response.json()
    if result.get("status") in {"ERROR", "MAINTENANCE"}:
        raise RuntimeError(f"MSA server rejected the request: {result}")
    ticket_id = result.get("id")
    if not ticket_id:
        raise RuntimeError(f"MSA server returned no ticket id: {result}")

    while result.get("status") in {"UNKNOWN", "RUNNING", "PENDING"}:
        check_deadline()
        logger.info("MSA ticket %s: %s", ticket_id, result.get("status"))
        time.sleep(poll_interval)
        response = _request_with_retries(
            "GET",
            f"{host_url.rstrip('/')}/ticket/{ticket_id}",
            headers=headers,
            auth=auth,
        )
        result = response.json()
    if result.get("status") != "COMPLETE":
        raise RuntimeError(f"MSA search did not complete: {result}")

    response = _request_with_retries(
        "GET",
        f"{host_url.rstrip('/')}/result/download/{ticket_id}",
        headers=headers,
        auth=auth,
    )
    temporary = archive_path.with_suffix(archive_path.suffix + ".partial")
    temporary.write_bytes(response.content)
    _safe_extract_tar(temporary, output_dir)
    temporary.replace(archive_path)


def _read_a3m(path: Path) -> tuple[list[str], list[str], int]:
    headers: list[str] = []
    sequences: list[str] = []
    uniref_boundary = 0
    query_header = ""
    for line_index, line in enumerate(path.read_text().splitlines(keepends=True)):
        if line.startswith(">"):
            headers.append(line)
            if not query_header:
                query_header = line
            elif line == query_header:
                uniref_boundary = line_index // 2
        else:
            sequences.append(line)
    if len(headers) != len(sequences):
        raise ValueError(
            f"Malformed A3M with {len(headers)} headers and {len(sequences)} sequences: {path}"
        )
    return headers, sequences, uniref_boundary


def _read_taxonomy(path: Path) -> dict[str, str]:
    mapping = {}
    for line in path.read_text().splitlines():
        fields = line.split("\t")
        if len(fields) >= 3:
            mapping[fields[1]] = fields[2]
    return mapping


def _write_processed_msas(
    query: str,
    raw_a3m: Path,
    output_dir: Path,
    taxonomy: Mapping[str, str] | None,
) -> None:
    headers, sequences, uniref_boundary = _read_a3m(raw_a3m)
    paired = [">query\n", f"{query}\n"]
    unpaired = [">query\n", f"{query}\n"]
    for index, (header, sequence) in enumerate(zip(headers, sequences)):
        if sequence.rstrip("\n") == query:
            continue
        aligned_length = sum(
            character.isupper() or character == "-" for character in sequence
        )
        if aligned_length != len(query):
            logger.warning(
                "Skipping malformed MSA row with %d aligned columns; expected %d",
                aligned_length,
                len(query),
            )
            continue
        identifier = header.split("\t", 1)[0].removeprefix(">").strip()
        taxid = taxonomy.get(identifier) if taxonomy else None
        if taxid is not None and index < uniref_boundary:
            if identifier.startswith("UniRef100_"):
                replacement = f"{identifier}_{taxid}/"
            else:
                replacement = f"UniRef100_{identifier}_{taxid}/"
            paired.extend([header.replace(identifier, replacement), sequence])
        else:
            unpaired.extend([header, sequence])
    (output_dir / "pairing.a3m").write_text("".join(paired))
    (output_dir / "non_pairing.a3m").write_text("".join(unpaired))


def first_a3m_query(path: Path) -> str:
    sequence_lines = []
    found_header = False
    for line in path.read_text().splitlines():
        if line.startswith(">"):
            if found_header:
                break
            found_header = True
        elif found_header:
            sequence_lines.append(line.strip())
    aligned = "".join(sequence_lines)
    return "".join(char for char in aligned if char.isupper() and char != "-")


def search_protein_msas(
    sequences: Sequence[str],
    output_dir: str | Path,
    *,
    host_url: str | None = None,
    email: str = "",
    poll_interval: int = 10,
    max_wait_seconds: int = 1800,
) -> list[Path]:
    """Search and postprocess one MSA directory per unique protein sequence."""
    if not sequences:
        return []
    if len(set(sequences)) != len(sequences):
        raise ValueError("search_protein_msas expects unique sequences")
    if poll_interval <= 0:
        raise ValueError("poll_interval must be positive")
    if max_wait_seconds <= 0:
        raise ValueError("max_wait_seconds must be positive")
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    server = host_url or os.environ.get("VENUSFOLD_MSA_SERVER_URL", DEFAULT_MSA_SERVER_URL)
    _submit_and_download(
        sequences,
        root,
        host_url=server,
        email=email,
        poll_interval=poll_interval,
        max_wait_seconds=max_wait_seconds,
    )

    taxonomy_path = root / "uniref_tax.m8"
    taxonomy = _read_taxonomy(taxonomy_path) if taxonomy_path.is_file() else None
    result_dirs = []
    for index, query in enumerate(sequences):
        result_dir = root / str(index)
        result_dir.mkdir(parents=True, exist_ok=True)
        raw_a3m = root / f"{index}.a3m"
        if raw_a3m.is_file():
            observed = first_a3m_query(raw_a3m)
            if observed != query:
                raise ValueError(
                    f"Raw MSA query mismatch in {raw_a3m}: expected sequence "
                    f"{index} with length {len(query)}, observed length "
                    f"{len(observed)}. Use a separate MSA cache directory."
                )
            _write_processed_msas(query, raw_a3m, result_dir, taxonomy)
        else:
            logger.warning("No MSA returned for sequence %d; using query only", index)
            query_only = f">query\n{query}\n"
            (result_dir / "pairing.a3m").write_text(query_only)
            (result_dir / "non_pairing.a3m").write_text(query_only)
        for filename in ("pairing.a3m", "non_pairing.a3m"):
            path = result_dir / filename
            observed = first_a3m_query(path)
            if observed != query:
                raise ValueError(
                    f"MSA query mismatch in {path}: expected length {len(query)}, "
                    f"observed length {len(observed)}"
                )
        result_dirs.append(result_dir)
    return result_dirs
