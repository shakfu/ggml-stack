# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1]

### Added
- `Makefile` task runner wrapping the `uv` dev loop (install, test, lint,
  format, typecheck, build, dist, clean).

## [0.1.0]

Initial standalone extraction of the shared single-ggml build recipe from the
near-identical `manage.py` builders in chimera, cyllama, and inferna.

### Added
- `build_stack()` / `StackConfig` / `stack_summary()` library API.
- Per-engine builders: `LlamaCppBuilder`, `WhisperCppBuilder`,
  `StableDiffusionCppBuilder`.
- Enforced build order (llama -> whisper -> sd) with one shared ggml: llama
  built with `-DGGML_MAX_NAME=128`, SD's vendored `ggml/` replaced by llama's
  checkout (`_sync_ggml_abi`) under `SD_USE_VENDORED_GGML=0`.
- CLI (`python -m ggml_stack`) with `build` and `info` subcommands, including
  `--dry-run`, `--engines`, `--with-server`, `--openssl`, `--no-shared-ggml`.
- Env-var parameterization preserving the consumers' CI recipes
  (`GGML_NATIVE`, `GGML_METAL/CUDA/VULKAN/HIP/SYCL/OPENCL`, version pins).
- Dry-run test suite asserting on generated git/cmake command lines.

[Unreleased]: https://github.com/ggml-org/ggml-stack/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ggml-org/ggml-stack/releases/tag/v0.1.0
