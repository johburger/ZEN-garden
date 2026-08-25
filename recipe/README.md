# Local conda recipe

This directory contains the initial conda-forge recipe for `zen-garden`.

The recipe currently uses `source.path: ..` so it can be tested against the
working tree that contains the dependency and CLI changes. Before opening the
conda-forge staged-recipes pull request:

1. Release a new ZEN-garden version containing these changes.
2. Set `version` in `recipe.yaml` to that released version.
3. Replace the local `source.path` with the PyPI source URL and the SHA256 hash
   of the new source distribution.
4. Verify that `johburger` is the correct GitHub username for the recipe
   maintainer, or replace it with the appropriate username.

For local validation, install `rattler-build` and run:

```bash
rattler-build build --recipe recipe/recipe.yaml
```

For the conda-forge-specific lint, use `conda smithy lint --conda-forge` from
the staged-recipes checkout after copying this recipe to
`recipes/zen-garden/recipe.yaml`.

The recipe uses `noarch: python` because ZEN-garden contains Python code only
and does not build a platform-specific extension.
