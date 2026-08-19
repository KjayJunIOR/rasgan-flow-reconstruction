# Contributing

Contributions should preserve the public data contract and distinguish clearly
between validated RASGAN behavior and experimental generator variants.

Contributions are expected to be submitted under the repository's academic
and non-commercial use terms unless a separate written agreement applies.

Before opening a pull request:

1. do not add private CFD data, checkpoints, or unpublished sponsor material;
2. do not copy upstream source without recording its license and provenance;
3. add tests for changes to array layout, checkpoint metadata, model shapes, or
   loss behavior;
4. run `pytest -m "not tensorflow"` and, when TensorFlow is available,
   `rasgan-selfcheck`; and
5. document any change that can make existing checkpoints incompatible.
