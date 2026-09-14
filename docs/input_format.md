# Input format

VenusFold accepts a JSON list. Each item has a unique `name` and a `sequences`
list. The supported molecule keys are `proteinChain`, `dnaSequence`,
`rnaSequence`, and `ligand`.

Minimal example without MSA or templates:

```json
[
  {
    "name": "example",
    "sequences": [
      {
        "proteinChain": {
          "sequence": "MSTNPKPQRKTKRNTNRRPQDVKFPGG",
          "count": 1
        }
      }
    ]
  }
]
```

For precomputed protein MSA/template inputs, add paths to the chain object:

```json
{
  "proteinChain": {
    "sequence": "MSTNPKPQRKTKRNTNRRPQDVKFPGG",
    "count": 1,
    "pairedMsaPath": "/path/to/pairing.a3m",
    "unpairedMsaPath": "/path/to/non_pairing.a3m",
    "templatesPath": "/path/to/hmmsearch.a3m"
  }
}
```

Paths should be absolute when inference is launched from a scheduler or another
working directory. When `use_msa` and `auto_search_msa` are both true (the
defaults), missing protein MSAs are searched automatically before inference.
Generated A3M files and the updated input JSON are cached below
`<dump_dir>/input_features/<input-hash>/`. Existing inline MSAs or valid
`pairedMsaPath`/`unpairedMsaPath` values are preserved. Use
`--msa_search_dir PATH` to choose another cache root or
`--auto_search_msa false` to disable online search.
When `use_template` is enabled, automatic preparation also adds missing
template-search results.

## Searching MSAs and templates

`scripts/prepare_msa_templates.py` accepts the minimal JSON format above. It
searches each unique protein sequence once, writes `pairing.a3m` and
`non_pairing.a3m`, and creates a new JSON containing their absolute paths.
Pass `--templates` to additionally run HMMER against PDB SeqRes and add
`templatesPath`.

```bash
python scripts/prepare_msa_templates.py \
  --input-json examples/input.json \
  --output-json output/example/input_msa_template.json \
  --msa-dir output/example/msa \
  --templates
```

The original JSON is not modified. Existing `out.tar.gz` server results in the
MSA directory are reused, making interrupted searches resumable. The generated
audit JSON records sequence hashes, lengths, MSA depths, template hit counts,
and result directories.

The service wait limit is 1800 seconds by default and can be changed with
`--max-wait-seconds`. Direct connections are used unless
`VENUSFOLD_MSA_USE_ENV_PROXY=true` is set. Downloaded archives are checked for
path traversal and link entries before extraction.
