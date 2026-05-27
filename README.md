# ggml-stack

The shared **single-ggml build recipe** for
[llama.cpp](https://github.com/ggml-org/llama.cpp) +
[whisper.cpp](https://github.com/ggml-org/whisper.cpp) +
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp),
factored out of the near-identical builder code in the `manage.py` files of
three sibling projects (chimera, cyllama, inferna).

All three build the three engines against **one** copy of ggml (rather than the
three separate vendored copies you get by default). That mechanism is small,
load-bearing, and drift-prone across upstream bumps -- and it was maintained
three times. This package holds it once. Each consuming project keeps its own
version pins, language bindings, and wheel packaging; it calls in here for the
fetch + compile + stage of the three engines.

> Status: standalone and tested in isolation (dry-run command-generation tests +
> a real end-to-end build). Not yet wired into the three projects -- that is a
> deliberate next step.

## The recipe

`build_stack()` enforces the order the mechanism requires:

1. **llama.cpp first.** It provides the ggml that the others link. When sharing
   is on, it is compiled with `-DGGML_MAX_NAME=128` (C and C++ flags) so the
   `ggml_tensor` struct layout matches what stable-diffusion.cpp needs.
2. **whisper.cpp.** Built normally; only `libwhisper.a` + `libcommon.a` are
   staged (whisper's own ggml is discarded -- the consumer links llama's).
3. **stable-diffusion.cpp.** With sharing on (`SD_USE_VENDORED_GGML=0`), SD's
   vendored `ggml/` subtree is replaced with llama.cpp's checkout
   (`_sync_ggml_abi`) before compiling, so SD's `ggml_op` / `ggml_type` enum ids
   match the ggml it links at runtime. Backends use SD's `SD_*` CMake names.

Output is staged per engine under `<root>/thirdparty/<engine>/{include,lib}`,
with sources under `<root>/build/<engine>/`.

## Install

```bash
uv sync                         # or: pip install -e .
uv run python -m ggml_stack --help
```

## Use

Library:

```python
from ggml_stack import build_stack, StackConfig, stack_summary

project = build_stack(StackConfig(
    root="/path/to/workdir",        # default: cwd
    engines=("llama", "whisper", "sd"),
    shared_ggml=True,               # SD links llama's ggml (the point)
    with_server=False,              # llama-server (server-context + cpp-httplib)
    openssl=False,                  # LLAMA_OPENSSL
))
print(stack_summary(project))       # {engine: [staged .a files]}
```

CLI:

```bash
# preview the exact git/cmake commands without running them:
python -m ggml_stack build --dry-run --root /tmp/work

# real build (host-tuned by default; set GGML_NATIVE=0 for portable):
python -m ggml_stack build --root /tmp/work --engines llama,whisper,sd

# chimera-style build with the embedded server:
python -m ggml_stack build --with-server --openssl

python -m ggml_stack info --root /tmp/work   # list staged archives
```

## Version pins

The three engines are pinned to git refs (tags or branch/commit names passed to
`git clone --branch`). The default pins live as constants near the top of
[`src/ggml_stack/builders.py`](src/ggml_stack/builders.py) (`LLAMACPP_VERSION` /
`WHISPERCPP_VERSION` / `SDCPP_VERSION`) -- that file is the source of truth for
the current values.

A pin can be set three ways, in increasing precedence:

1. **Default constant** -- the literal fallback in `builders.py`.
2. **Environment variable** -- `LLAMACPP_VERSION` / `WHISPERCPP_VERSION` /
   `SDCPP_VERSION`, read at import time.
3. **`StackConfig.versions`** -- a `dict` keyed by `"llama"` / `"whisper"` /
   `"sd"`, applied per-builder at `build_stack()` time. This is the per-consumer
   override each project uses to keep its own pins.

So `StackConfig.versions` > env var > default constant. Example:

```python
build_stack(StackConfig(versions={"llama": "b9400", "sd": "master-660-abc1234"}))
```

These pins are independent of the `ggml-stack` package version in
`pyproject.toml`, which versions this package, not the engines it builds.

## Parameterization (what differs between the three consumers)

| Knob | How | chimera | cyllama / inferna |
|---|---|---|---|
| Version pins | `StackConfig.versions` or `LLAMACPP_VERSION` etc. env | own pins | own pins |
| llama-server | `with_server` | on | off |
| OpenSSL | `openssl` | env-gated, default off | on |
| CPU tuning | `GGML_NATIVE` env (`1`/`0`) | host (`1`) | portable wheels (`0`) |
| Backend | `GGML_METAL/CUDA/VULKAN/HIP/SYCL/OPENCL` env | one per install | matrix |
| Shared ggml | `shared_ggml` / `SD_USE_VENDORED_GGML` | on | on |

Backend selection, CUDA/HIP arch lists, OpenMP, and `GGML_CPU_ALL_VARIANTS` are
all read from the same env vars the source projects use, so existing CI recipes
keep working unchanged.

## What is intentionally NOT here

Project-specific concerns stay in each project's `manage.py`: Cython/nanobind
extension builds, wheel packaging, cibuildwheel, `auditwheel`/`delocate`/
`delvewheel` repair, the webui embedding, sqlite / sqlite-vec vendoring, archive
combining (`combine_archives.py`), and the CLI subcommands. ggml-stack is only
the fetch + single-ggml compile + stage of the three engines.

## ABI notes (read before integrating)

- **`GGML_MAX_NAME` for whisper.** This recipe applies `-DGGML_MAX_NAME=128`
  to llama (and thus to SD via the ggml sync), matching the source projects,
  but does **not** apply it to whisper -- also matching them. whisper is built
  with its default and then linked against llama's `GGML_MAX_NAME=128` ggml.
  The three projects ship this way and pass their tests, so it is reproduced
  faithfully; but it is the most likely latent ABI hazard if a future whisper
  pin starts depending on the struct layout. Flagged here so integration can
  verify rather than assume.
- The combined-archive whole-archive contract (ggml backends must be
  force-loaded so their static initializers run) lives in the consumer, not
  here -- ggml-stack only produces the per-engine archives.

## Tests

```bash
uv run --extra test python -m pytest -q
```

The tests run in dry-run mode (no toolchain needed): they assert on the
generated git/cmake command lines -- build order, the `GGML_MAX_NAME`
propagation, `LLAMA_BUILD_SERVER`/`LLAMA_OPENSSL`, `SD_*` backend names, and the
SD ggml sync.
