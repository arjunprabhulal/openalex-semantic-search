# Contributing

Thanks for your interest. Bug reports, benchmark results, labelled queries and focused
pull requests are all welcome.

## Before you start

- **Bugs:** open an issue with the command you ran, the stage (10k, 1m, full), your
  platform, and the full error.
- **Features or design changes:** open an issue first so we can agree on the approach
  before you write code.
- **Security issues:** do not open an issue. Follow [SECURITY.md](SECURITY.md).

## Development setup

```bash
git clone https://github.com/arjunprabhulal/openalex-semantic-search
cd openalex-semantic-search
python -m venv .venv && source .venv/bin/activate
pip install -e '.[runtime,test]'
pre-commit install          # optional: runs the secret scan before each commit

# macOS only: raise the open-file limit and avoid an OpenMP clash
ulimit -n 2048 && export OMP_NUM_THREADS=1

python -m pytest
```

The test suite needs no network, GPU or model download; it uses a deterministic hashing
embedder and a NumPy index.

## Pull requests

- Keep each pull request to one change, with tests.
- Retrieval or ranking changes need numbers: run `openalex-semantic-search benchmark` on
  at least a 10K generation, before and after, and paste the results.
- Do not loosen a quality gate to make a change pass. A failed gate is evidence.
- Never commit keys, `.env` files, indexes, artifacts or model caches.
- Update `README.md` when behaviour, flags or API fields change.

## Licence of contributions

By contributing, you agree that your contributions are licensed under the
[Apache License 2.0](LICENSE), as described in its Section 5.
