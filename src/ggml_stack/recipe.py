"""High-level orchestration: build the three engines against one shared ggml.

`build_stack` is the single entry point a consuming project's manage.py would
call. It enforces the build ORDER that the single-ggml mechanism requires:
llama.cpp first (it owns the ggml that the others link), then whisper, then SD
(whose source ggml is replaced with llama's before it compiles).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .builders import (
    LlamaCppBuilder,
    StableDiffusionCppBuilder,
    WhisperCppBuilder,
    shared_ggml_enabled,
)
from .project import Project

ENGINES = ("llama", "whisper", "sd")


@dataclass
class StackConfig:
    """Knobs for a single-ggml build. Defaults match the common (non-server) case."""

    root: Path | None = None
    engines: tuple[str, ...] = ENGINES
    shared_ggml: bool = True       # SD links llama's ggml (the whole point)
    with_server: bool = False      # llama-server (server-context + cpp-httplib)
    openssl: bool = False          # LLAMA_OPENSSL for cpp-httplib HTTPS
    sd_examples: bool = False
    versions: dict[str, str] = field(default_factory=dict)  # engine -> pin override
    dry_run: bool = False


def build_stack(config: StackConfig | None = None) -> Project:
    """Fetch + build the configured engines against one shared ggml.

    Returns the Project so the caller can locate staged output under
    project.thirdparty/<engine>/{include,lib}.
    """
    cfg = config or StackConfig()

    # The shared-ggml decision is communicated to the builders via the same
    # env var the source projects use, so the wiring stays identical.
    if cfg.shared_ggml:
        os.environ["SD_USE_VENDORED_GGML"] = "0"
    elif os.environ.get("SD_USE_VENDORED_GGML") == "0":
        del os.environ["SD_USE_VENDORED_GGML"]

    project = Project(root=cfg.root, dry_run=cfg.dry_run)
    project.setup()

    # llama.cpp MUST be built first: it provides the ggml that whisper/sd link,
    # and SD's _sync_ggml_abi copies from llama's checkout.
    if "llama" in cfg.engines:
        LlamaCppBuilder(
            version=cfg.versions.get("llama"),
            project=project,
            with_server=cfg.with_server,
            openssl=cfg.openssl,
        ).build()

    if "whisper" in cfg.engines:
        WhisperCppBuilder(
            version=cfg.versions.get("whisper"), project=project
        ).build()

    if "sd" in cfg.engines:
        if cfg.shared_ggml and "llama" not in cfg.engines:
            raise ValueError(
                "shared_ggml requires building 'llama' too "
                "(SD links/syncs llama.cpp's ggml)"
            )
        StableDiffusionCppBuilder(
            version=cfg.versions.get("sd"), project=project
        ).build(examples=cfg.sd_examples)

    return project


def stack_summary(project: Project) -> dict[str, list[str]]:
    """Map each staged engine to the static archives present under its lib/."""
    out: dict[str, list[str]] = {}
    for engine_dir in sorted(project.thirdparty.glob("*")):
        lib = engine_dir / "lib"
        if lib.is_dir():
            out[engine_dir.name] = sorted(p.name for p in lib.glob("*.a")) or sorted(
                p.name for p in lib.glob("*.lib")
            )
    return out


__all__ = [
    "ENGINES",
    "StackConfig",
    "build_stack",
    "stack_summary",
    "shared_ggml_enabled",
]
