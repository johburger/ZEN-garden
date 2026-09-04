# Conda-forge recipe

This directory contains the conda-forge recipe for `zen-garden` v3.0.0.
The recipe builds the released PyPI source distribution.

For local validation from the ZEN-garden repository root, install
`rattler-build` and run:

```bash
rattler-build build --recipe recipe/recipe.yaml
```

For the conda-forge-specific lint, copy this file to
`recipes/zen-garden/recipe.yaml` in your staged-recipes checkout and run from
that checkout's root:

```bash
conda-smithy recipe-lint --conda-forge recipes/zen-garden
```

The staged-recipes checkout root is the directory printed by
`git rev-parse --show-toplevel` when run inside that checkout.

The recipe uses `noarch: python` because ZEN-garden contains Python code only
and does not build a platform-specific extension.
