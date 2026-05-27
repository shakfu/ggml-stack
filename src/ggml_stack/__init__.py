"""ggml-stack: the shared single-ggml build recipe for llama.cpp +
whisper.cpp + stable-diffusion.cpp.

Factored out of the (near-identical) builder code in the chimera / cyllama /
inferna `manage.py` files so the load-bearing single-ggml mechanism lives in
one place. Consuming projects keep their own version pins, bindings, and wheel
packaging; they call into the builders / `build_stack` here for the fetch +
compile + stage of the three engines against one ggml.
"""

from __future__ import annotations

from .builders import (
    GGML_MAX_NAME,
    AbstractBuilder,
    Builder,
    GgmlBuilder,
    LlamaCppBuilder,
    StableDiffusionCppBuilder,
    WhisperCppBuilder,
    shared_ggml_enabled,
)
from .project import Project
from .recipe import ENGINES, StackConfig, build_stack, stack_summary
from .shell import ShellCmd, getenv, setenv, setup_logging

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # shell / project
    "ShellCmd",
    "Project",
    "getenv",
    "setenv",
    "setup_logging",
    # builders
    "AbstractBuilder",
    "Builder",
    "GgmlBuilder",
    "LlamaCppBuilder",
    "WhisperCppBuilder",
    "StableDiffusionCppBuilder",
    "GGML_MAX_NAME",
    "shared_ggml_enabled",
    # recipe
    "ENGINES",
    "StackConfig",
    "build_stack",
    "stack_summary",
]
