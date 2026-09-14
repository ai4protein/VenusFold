from __future__ import annotations

import json
from pathlib import Path

from venusfold.data.msa import pipeline


def _write_query_msa(directory: Path, sequence: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    content = f">query\n{sequence}\n"
    (directory / "pairing.a3m").write_text(content)
    (directory / "non_pairing.a3m").write_text(content)


def test_prepare_input_json_searches_unique_missing_sequences(
    tmp_path: Path, monkeypatch
) -> None:
    sequence = "ABCDE"
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            [
                {
                    "name": "first",
                    "sequences": [
                        {"proteinChain": {"sequence": sequence, "count": 1}}
                    ],
                },
                {
                    "name": "second",
                    "sequences": [
                        {"proteinChain": {"sequence": sequence, "count": 1}}
                    ],
                },
            ]
        )
    )
    calls = []

    def fake_search(sequences, output_dir, **kwargs):
        calls.append(list(sequences))
        results = []
        for index, item in enumerate(sequences):
            result = Path(output_dir) / str(index)
            _write_query_msa(result, item)
            results.append(result)
        return results

    monkeypatch.setattr(pipeline, "search_protein_msas", fake_search)
    output = tmp_path / "prepared.json"
    prepared = pipeline.prepare_input_json(source, output, tmp_path / "msa")

    assert prepared == output.resolve()
    assert calls == [[sequence]]
    jobs = json.loads(output.read_text())
    first = jobs[0]["sequences"][0]["proteinChain"]
    second = jobs[1]["sequences"][0]["proteinChain"]
    assert first["pairedMsaPath"] == second["pairedMsaPath"]
    assert Path(first["unpairedMsaPath"]).is_file()
    audit = json.loads(
        (tmp_path / "prepared_msa_template_audit.json").read_text()
    )
    assert len(audit["searched_sequences"]) == 1


def test_prepare_input_json_preserves_existing_msa(tmp_path: Path, monkeypatch) -> None:
    sequence = "ABCDE"
    existing = tmp_path / "existing.a3m"
    existing.write_text(f">query\n{sequence}\n")
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            [
                {
                    "name": "existing",
                    "sequences": [
                        {
                            "proteinChain": {
                                "sequence": sequence,
                                "count": 1,
                                "unpairedMsaPath": str(existing),
                            }
                        }
                    ],
                }
            ]
        )
    )

    def unexpected_search(*args, **kwargs):
        raise AssertionError("search should not run for a valid existing MSA")

    monkeypatch.setattr(pipeline, "search_protein_msas", unexpected_search)
    prepared = pipeline.prepare_input_json(
        source, tmp_path / "prepared.json", tmp_path / "msa"
    )

    assert prepared == source.resolve()


def test_prepare_input_json_replaces_mismatched_existing_msa(
    tmp_path: Path, monkeypatch
) -> None:
    sequence = "ABCDE"
    mismatched = tmp_path / "mismatched.a3m"
    mismatched.write_text(">query\nWRONG\n")
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            [
                {
                    "name": "mismatch",
                    "sequences": [
                        {
                            "proteinChain": {
                                "sequence": sequence,
                                "count": 1,
                                "unpairedMsaPath": str(mismatched),
                            }
                        }
                    ],
                }
            ]
        )
    )

    def fake_search(sequences, output_dir, **kwargs):
        result = Path(output_dir) / "0"
        _write_query_msa(result, sequences[0])
        return [result]

    monkeypatch.setattr(pipeline, "search_protein_msas", fake_search)
    output = tmp_path / "prepared.json"
    prepared = pipeline.prepare_input_json(source, output, tmp_path / "msa")

    assert prepared == output.resolve()
    chain = json.loads(output.read_text())[0]["sequences"][0]["proteinChain"]
    assert chain["unpairedMsaPath"] != str(mismatched)
    assert pipeline.first_a3m_query(Path(chain["unpairedMsaPath"])) == sequence


def test_prepare_input_json_reuses_matching_generated_output(
    tmp_path: Path, monkeypatch
) -> None:
    sequence = "ABCDE"
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            [
                {
                    "name": "cached",
                    "sequences": [
                        {"proteinChain": {"sequence": sequence, "count": 1}}
                    ],
                }
            ]
        )
    )
    call_count = 0

    def fake_search(sequences, output_dir, **kwargs):
        nonlocal call_count
        call_count += 1
        result = Path(output_dir) / "0"
        _write_query_msa(result, sequences[0])
        return [result]

    monkeypatch.setattr(pipeline, "search_protein_msas", fake_search)
    output = tmp_path / "prepared.json"
    pipeline.prepare_input_json(source, output, tmp_path / "msa")
    pipeline.prepare_input_json(source, output, tmp_path / "msa")

    assert call_count == 1


def test_prepare_input_json_reuses_same_sequence_from_input(
    tmp_path: Path, monkeypatch
) -> None:
    sequence = "ABCDE"
    existing = tmp_path / "existing.a3m"
    existing.write_text(f">query\n{sequence}\n")
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            [
                {
                    "name": "reuse",
                    "sequences": [
                        {
                            "proteinChain": {
                                "sequence": sequence,
                                "count": 1,
                                "unpairedMsaPath": str(existing),
                            }
                        },
                        {"proteinChain": {"sequence": sequence, "count": 1}},
                    ],
                }
            ]
        )
    )

    def unexpected_search(*args, **kwargs):
        raise AssertionError("an MSA for the same sequence should be reused")

    monkeypatch.setattr(pipeline, "search_protein_msas", unexpected_search)
    output = tmp_path / "prepared.json"
    prepared = pipeline.prepare_input_json(source, output, tmp_path / "msa")

    assert prepared == output.resolve()
    jobs = json.loads(output.read_text())
    reused = jobs[0]["sequences"][1]["proteinChain"]
    assert reused["unpairedMsaPath"] == str(existing.resolve())


def test_template_request_upgrades_msa_only_cache(tmp_path: Path, monkeypatch) -> None:
    sequence = "ABCDE"
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            [
                {
                    "name": "template_upgrade",
                    "sequences": [
                        {"proteinChain": {"sequence": sequence, "count": 1}}
                    ],
                }
            ]
        )
    )
    search_calls = 0
    template_calls = 0

    def fake_search(sequences, output_dir, **kwargs):
        nonlocal search_calls
        search_calls += 1
        result = Path(output_dir) / "0"
        _write_query_msa(result, sequences[0])
        return [result]

    def fake_template(msa_dir, **kwargs):
        nonlocal template_calls
        template_calls += 1
        output = Path(msa_dir) / "hmmsearch.a3m"
        output.write_text(">template\nACDE\n")
        return output

    monkeypatch.setattr(pipeline, "search_protein_msas", fake_search)
    monkeypatch.setattr(pipeline, "search_templates", fake_template)
    output = tmp_path / "prepared.json"
    pipeline.prepare_input_json(source, output, tmp_path / "msa")
    pipeline.prepare_input_json(
        source,
        output,
        tmp_path / "msa",
        include_templates=True,
    )

    assert search_calls == 2
    assert template_calls == 1
    chain = json.loads(output.read_text())[0]["sequences"][0]["proteinChain"]
    assert Path(chain["templatesPath"]).is_file()


def test_legacy_audit_format_is_not_reused(tmp_path: Path, monkeypatch) -> None:
    sequence = "ABCDE"
    source = tmp_path / "input.json"
    source.write_text(
        json.dumps(
            [
                {
                    "name": "legacy_audit",
                    "sequences": [
                        {"proteinChain": {"sequence": sequence, "count": 1}}
                    ],
                }
            ]
        )
    )
    calls = 0

    def fake_search(sequences, output_dir, **kwargs):
        nonlocal calls
        calls += 1
        result = Path(output_dir) / "0"
        _write_query_msa(result, sequences[0])
        return [result]

    monkeypatch.setattr(pipeline, "search_protein_msas", fake_search)
    output = tmp_path / "prepared.json"
    pipeline.prepare_input_json(source, output, tmp_path / "msa")
    audit = tmp_path / "prepared_msa_template_audit.json"
    audit.write_text("[]\n")
    pipeline.prepare_input_json(source, output, tmp_path / "msa")

    assert calls == 2
    assert isinstance(json.loads(audit.read_text()), dict)
