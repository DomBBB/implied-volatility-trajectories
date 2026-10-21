# Implied Volatility Trajectories

Research code for modelling and forecasting the evolution of implied volatility
surfaces, including probabilistic surface trajectories, joint surface–return
models, and economic admissibility.

The repository is organised by research project. Each project directory contains
its own implementation, documentation, experimental details, and publication
references, allowing related work to develop alongside the original studies.

## Projects

| Directory | Research focus | Publication |
| :-- | :-- | :-- |
| [ICAIF26](ICAIF26/) | Joint 30-day SPX implied volatility surface and return trajectories using conditional latent diffusion | [Buchegger and Gonon (2026)](https://arxiv.org/abs/2608.22478), accepted at ICAIF 2026 |

### ICAIF26

**[Arbitrage-Aware Probabilistic Multi-Step Forecasting of Implied Volatility Surfaces: Joint Generation of Surface-Return Trajectories Using Latent Diffusion](https://arxiv.org/abs/2608.22478)**

The project combines an arbitrage-aware surface autoencoder with a conditional
diffusion model for joint surface–return forecasting.

[Project README, results, and setup](ICAIF26/README.md) · [Notebook](ICAIF26/ICAIF.ipynb) · [Paper](https://arxiv.org/abs/2608.22478)

## Using the repository

Start with the README in the relevant project directory. Data requirements,
dependencies, available artifacts, and execution instructions are documented
for each project.

## Citation

Please cite the publication associated with the project you use. Citation details are provided in
the corresponding project README.

## License

Project source code is provided under the [MIT License](LICENSE). Bundled
third-party components retain their respective attribution and license notices.
