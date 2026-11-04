# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project uses
[Semantic Versioning](https://semver.org/).

## [1.1.0] - 2026-11-04

### Added
- Full second-order Sobol analysis: an independent second-order design
  (`calc_second_order=true`, 6400 runs) with resumable simulation across all
  operating conditions, enabling pairwise S2 indices.
- Project tooling: `pre-commit` config, Dependabot, issue templates, and a pull
  request template.

### Changed
- Migrated sampling from the deprecated `SALib.sample.saltelli` entry to
  `SALib.sample.sobol`; deprecation warnings are now eliminated.
- CLI tests rewritten to invoke `main()` directly and run `cmd_sample` against a
  real temporary configuration, improving CLI coverage.

## [1.0.0]

### Added
- PyBaMM P2D/DFN model wrapper with 24 parameter specifications across six
  parameter groups.
- Morris and Sobol global sensitivity analysis with resumable design/checkpoint
  framework and solver-failure handling.
- Total-effect (ST) heatmaps, parameter whitelist, and ST–S1 interaction
  analysis with share breakdowns.
- CLI, reporting, PEP 621 packaging, Ruff linting, pytest suite, and CI.
