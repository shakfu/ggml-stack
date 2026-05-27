# TODO

Working list for ggml-stack. Items roughly ordered by priority within each
section. See `README.md` for the design rationale behind several of these.

## Integration (the deliberate next step)

- [ ] Wire the recipe into the three consumers, replacing their `manage.py`
      builder code: chimera, cyllama, inferna.
- [ ] Confirm each consumer keeps owning version pins, language bindings, and
      wheel packaging; ggml-stack stays fetch + compile + stage only.
- [ ] Validate the combined-archive whole-archive contract still holds in each
      consumer (ggml backends force-loaded so static initializers run) -- this
      lives in the consumer, not here.

## ABI / correctness hazards

- [ ] Verify the whisper `GGML_MAX_NAME` assumption on the next whisper pin:
      whisper is built with its default and linked against llama's
      `GGML_MAX_NAME=128` ggml. Flagged as the most likely latent ABI hazard.
- [ ] Add a check (or documented manual step) that SD's `ggml_op` / `ggml_type`
      enum ids match the synced ggml after an upstream bump.

## Testing

- [ ] Add a real end-to-end build test gated behind a marker / env flag so it
      stays opt-in (current suite is dry-run only, no toolchain needed).
- [ ] Cover `--no-shared-ggml` (SD uses its own vendored ggml) in the dry-run
      assertions.

## Tooling

- [ ] Decide ruff/mypy strategy: pin them in a `dev` extra and switch the
      Makefile `lint`/`format`/`typecheck` targets from `uvx` to `uv run`, or
      keep them unpinned via `uvx`.
- [ ] Add CI (lint + dry-run tests) once integration targets are settled.

## Docs

- [ ] Document per-consumer integration steps once the first wiring lands.
