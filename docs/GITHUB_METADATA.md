# Suggested GitHub metadata

## Repository name

```text
rasgan-flow-reconstruction
```

## About description

```text
TensorFlow conditional relativistic GAN for POD-to-reference flow-field reconstruction, with RRDB/ECA, spectral and physics-aware losses, EMA, and synthetic demo data.
```

## Topics

```text
tensorflow
generative-adversarial-network
scientific-machine-learning
computational-fluid-dynamics
flow-field-reconstruction
super-resolution
proper-orthogonal-decomposition
spectral-loss
physics-informed-machine-learning
computer-vision
```

## Profile pin summary

The README is structured so the pinned-repository card leads immediately to:

1. the problem and three physical channels;
2. the synthetic paired-field preview;
3. the custom generator, discriminator, and loss system;
4. a preliminary research visualization; and
5. runnable preprocessing, self-check, training, and inference commands.

Before publishing, add the final repository URLs to `pyproject.toml`, retain
the academic/non-commercial use terms and upstream attribution, and verify the
TensorFlow smoke test in the target environment.
