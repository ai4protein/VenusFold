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
working directory. Set `--use_msa false` and `--use_template false` when these
features are not supplied.

