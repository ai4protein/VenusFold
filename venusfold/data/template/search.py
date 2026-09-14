"""Template search from protein MSAs using HMMER."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import requests

from venusfold.data.tools.search import HmmsearchConfig, run_hmmsearch_with_a3m


logger = logging.getLogger(__name__)
DEFAULT_SEQRES_URL = (
    "https://protenix.tos-cn-beijing.volces.com/search_database/"
    "pdb_seqres_2022_09_28.fasta"
)


def _resolve_binary(value: str | None, name: str) -> str:
    path = value or shutil.which(name)
    if path is None or not Path(path).is_file():
        raise FileNotFoundError(
            f"{name} was not found. Install HMMER or pass --{name}-binary."
        )
    return path


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    logger.info("Downloading template database from %s", url)
    with requests.Session() as session:
        session.trust_env = os.environ.get(
            "VENUSFOLD_MSA_USE_ENV_PROXY", "false"
        ).lower() in {"1", "true", "yes"}
        with session.get(url, stream=True, timeout=(10, 300)) as response:
            response.raise_for_status()
            with temporary.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
    temporary.replace(destination)


def default_seqres_database() -> Path:
    root = Path(os.environ.get("VENUSFOLD_ROOT_DIR", str(Path.home())))
    return root / "search_database" / "pdb_seqres_2022_09_28.fasta"


def search_templates(
    msa_dir: str | Path,
    *,
    hmmsearch_binary: str | None = None,
    hmmbuild_binary: str | None = None,
    seqres_database: str | Path | None = None,
    seqres_url: str = DEFAULT_SEQRES_URL,
    msa_names: tuple[str, ...] = ("pairing", "non_pairing"),
) -> Path:
    """Write ``hmmsearch.a3m`` beside paired/unpaired protein MSAs."""
    result_dir = Path(msa_dir).resolve()
    hmmsearch = _resolve_binary(hmmsearch_binary, "hmmsearch")
    hmmbuild = _resolve_binary(hmmbuild_binary, "hmmbuild")
    database = Path(seqres_database).resolve() if seqres_database else default_seqres_database()
    if not database.is_file():
        _download(seqres_url, database)

    chunks = []
    for name in msa_names:
        path = result_dir / f"{name}.a3m"
        if path.is_file():
            text = path.read_text()
            chunks.append(text if text.endswith("\n") else text + "\n")
    if not chunks:
        raise FileNotFoundError(f"No input A3M files found in {result_dir}")

    config = HmmsearchConfig(
        hmmsearch_binary_path=hmmsearch,
        hmmbuild_binary_path=hmmbuild,
        filter_f1=0.1,
        filter_f2=0.1,
        filter_f3=0.1,
        e_value=100,
        inc_e=100,
        dom_e=100,
        incdom_e=100,
        alphabet="amino",
    )
    result = run_hmmsearch_with_a3m(
        database_path=str(database),
        hmmsearch_config=config,
        max_a3m_query_sequences=300,
        a3m="".join(chunks),
    )
    output = result_dir / "hmmsearch.a3m"
    output.write_text(result)
    if output.stat().st_size == 0:
        raise RuntimeError(f"Template search produced an empty file: {output}")
    logger.info("Template search result: %s", output)
    return output
