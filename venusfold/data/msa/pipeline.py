"""Prepare inference JSON files with searched protein MSAs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator

from venusfold.data.msa.search import first_a3m_query, search_protein_msas
from venusfold.data.template.search import search_templates


def sequence_digest(sequence: str) -> str:
    return hashlib.sha256(sequence.encode()).hexdigest()[:16]


def count_a3m_rows(path: Path) -> int:
    return sum(line.startswith(">") for line in path.read_text().splitlines())


def _existing_msa(chain: dict[str, Any]) -> dict[str, Any] | None:
    query = chain.get("sequence")
    inline = {
        key: chain[key]
        for key in ("pairedMsa", "unpairedMsa")
        if chain.get(key)
    }
    if inline:
        if _has_template(chain):
            inline["templatesPath"] = str(Path(chain["templatesPath"]).resolve())
        return inline

    supplied_paths = {
        key: Path(chain[key])
        for key in ("pairedMsaPath", "unpairedMsaPath")
        if chain.get(key)
    }
    if supplied_paths and all(
        path.is_file() and query and first_a3m_query(path) == query
        for path in supplied_paths.values()
    ):
        paths = {key: str(path.resolve()) for key, path in supplied_paths.items()}
        if _has_template(chain):
            paths["templatesPath"] = str(Path(chain["templatesPath"]).resolve())
        return paths

    old_msa = chain.get("msa")
    if isinstance(old_msa, dict) and old_msa.get("precomputed_msa_dir"):
        msa_dir = Path(old_msa["precomputed_msa_dir"])
        old_paths = [
            msa_dir / name
            for name in ("pairing.a3m", "non_pairing.a3m")
            if (msa_dir / name).is_file()
        ]
        if old_paths and all(
            query and first_a3m_query(path) == query for path in old_paths
        ):
            values = {"msa": old_msa}
            if _has_template(chain):
                values["templatesPath"] = str(Path(chain["templatesPath"]).resolve())
            return values
    return None


def _has_template(chain: dict[str, Any]) -> bool:
    path = chain.get("templatesPath")
    return bool(path and Path(path).is_file())


def _msa_directory(values: dict[str, Any]) -> Path | None:
    for key in ("pairedMsaPath", "unpairedMsaPath"):
        if values.get(key):
            return Path(values[key]).resolve().parent
    old_msa = values.get("msa")
    if isinstance(old_msa, dict) and old_msa.get("precomputed_msa_dir"):
        return Path(old_msa["precomputed_msa_dir"]).resolve()
    return None


def _protein_chains(jobs: list[dict[str, Any]]) -> Iterator[dict[str, Any]]:
    for job in jobs:
        for entity in job.get("sequences", []):
            chain = entity.get("proteinChain")
            if chain is not None:
                yield chain


def _load_jobs(path: Path) -> list[dict[str, Any]]:
    jobs = json.loads(path.read_text())
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Input JSON must contain a non-empty list of inference jobs")
    return jobs


def _file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_input_json(
    input_json: str | Path,
    output_json: str | Path,
    msa_dir: str | Path,
    *,
    host_url: str | None = None,
    email: str = "",
    poll_interval: int = 10,
    max_wait_seconds: int = 1800,
    include_templates: bool = False,
    hmmsearch_binary: str | None = None,
    hmmbuild_binary: str | None = None,
    seqres_database: str | Path | None = None,
) -> Path:
    """Search missing protein MSAs and return the prepared inference JSON."""
    source_path = Path(input_json).resolve()
    destination = Path(output_json).resolve()
    jobs = _load_jobs(source_path)
    source_digest = _file_digest(source_path)
    audit_path = destination.with_name(destination.stem + "_msa_template_audit.json")

    if destination.is_file() and audit_path.is_file():
        cached_jobs = _load_jobs(destination)
        cached_audit = json.loads(audit_path.read_text())
        if (
            isinstance(cached_audit, dict)
            and cached_audit.get("input_sha256") == source_digest
            and all(_existing_msa(chain) for chain in _protein_chains(cached_jobs))
            and (
                not include_templates
                or (
                    cached_audit.get("include_templates") is True
                    and all(
                        _has_template(chain) for chain in _protein_chains(cached_jobs)
                    )
                )
            )
        ):
            return destination

    known_by_sequence: dict[str, dict[str, Any]] = {}
    missing_sequences = set()
    for chain in _protein_chains(jobs):
        sequence = chain["sequence"]
        existing = _existing_msa(chain)
        if existing:
            known_by_sequence.setdefault(sequence, existing)
        else:
            missing_sequences.add(sequence)
    missing_sequences.difference_update(known_by_sequence)

    searched_by_sequence: dict[str, dict[str, str]] = {}
    audit = []
    sequences = sorted(missing_sequences)
    if sequences:
        result_dirs = search_protein_msas(
            sequences,
            msa_dir,
            host_url=host_url,
            email=email,
            poll_interval=poll_interval,
            max_wait_seconds=max_wait_seconds,
        )
        if len(result_dirs) != len(sequences):
            raise RuntimeError(
                f"MSA result count mismatch: {len(result_dirs)} != {len(sequences)}"
            )
        for sequence, result_dir in zip(sequences, result_dirs):
            paired = result_dir / "pairing.a3m"
            unpaired = result_dir / "non_pairing.a3m"
            for path in (paired, unpaired):
                if first_a3m_query(path) != sequence:
                    raise ValueError(f"MSA query does not match input sequence: {path}")
            values = {
                "pairedMsaPath": str(paired.resolve()),
                "unpairedMsaPath": str(unpaired.resolve()),
            }
            template = None
            if include_templates:
                template = search_templates(
                    result_dir,
                    hmmsearch_binary=hmmsearch_binary,
                    hmmbuild_binary=hmmbuild_binary,
                    seqres_database=seqres_database,
                )
                values["templatesPath"] = str(template.resolve())
            searched_by_sequence[sequence] = values
            audit.append(
                {
                    "sequence_sha256_16": sequence_digest(sequence),
                    "length": len(sequence),
                    "pairing_rows": count_a3m_rows(paired),
                    "non_pairing_rows": count_a3m_rows(unpaired),
                    "template_rows": count_a3m_rows(template) if template else None,
                    "result_dir": str(result_dir.resolve()),
                }
            )

    if include_templates:
        for sequence, values in known_by_sequence.items():
            if values.get("templatesPath"):
                continue
            existing_msa_dir = _msa_directory(values)
            if existing_msa_dir is None:
                raise ValueError(
                    "Template search requires file-backed MSA paths for existing "
                    f"sequence {sequence_digest(sequence)}"
                )
            template = search_templates(
                existing_msa_dir,
                hmmsearch_binary=hmmsearch_binary,
                hmmbuild_binary=hmmbuild_binary,
                seqres_database=seqres_database,
            )
            values["templatesPath"] = str(template.resolve())

    updated = False
    for chain in _protein_chains(jobs):
        if _existing_msa(chain):
            if include_templates and not _has_template(chain):
                template_path = known_by_sequence[chain["sequence"]].get(
                    "templatesPath"
                )
                if template_path:
                    chain["templatesPath"] = template_path
                    updated = True
            continue
        sequence = chain["sequence"]
        values = known_by_sequence.get(sequence) or searched_by_sequence.get(sequence)
        if values is None:
            raise RuntimeError(f"No MSA was prepared for sequence {sequence_digest(sequence)}")
        chain.update(values)
        updated = True

    if not updated:
        return source_path

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    temporary.write_text(json.dumps(jobs, indent=2) + "\n")
    temporary.replace(destination)
    audit_path.write_text(
        json.dumps(
            {
                "input_sha256": source_digest,
                "include_templates": include_templates,
                "searched_sequences": audit,
            },
            indent=2,
        )
        + "\n"
    )
    return destination
