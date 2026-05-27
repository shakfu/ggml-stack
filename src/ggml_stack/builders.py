"""Engine builders for the single-ggml stack.

Extracted from the (near-identical) builder hierarchies in chimera / cyllama /
inferna. The class shape is preserved:

    ShellCmd -> AbstractBuilder -> Builder -> GgmlBuilder -> {Llama,Whisper,SD}

The pieces that differ between the three consumers are turned into constructor
parameters rather than hard-coded (version pins, the llama target/lib set,
LLAMA_BUILD_SERVER, LLAMA_OPENSSL, whether to share ggml). The single-ggml
mechanism itself -- the load-bearing, drift-prone part -- is identical and lives
here once.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .project import Project
from .shell import PLATFORM, ShellCmd, forward_env_flags, getenv

# --------------------------------------------------------------------------
# Default version pins. Consumers override per-builder (constructor `version=`)
# or via env (LLAMACPP_VERSION / WHISPERCPP_VERSION / SDCPP_VERSION). These
# defaults track chimera's current pins; they are deliberately not the single
# source of truth -- each consuming project keeps its own.
# --------------------------------------------------------------------------
LLAMACPP_VERSION = os.environ.get("LLAMACPP_VERSION", "b9318")
WHISPERCPP_VERSION = os.environ.get("WHISPERCPP_VERSION", "v1.8.4")
SDCPP_VERSION = os.environ.get("SDCPP_VERSION", "master-650-1ceb5bd")

# stable-diffusion.cpp needs long tensor names. When SD shares llama.cpp's ggml
# both sides must agree on this or the ggml_tensor struct layout diverges.
GGML_MAX_NAME = 128


def shared_ggml_enabled() -> bool:
    """True when SD is configured to link llama.cpp's ggml (SD_USE_VENDORED_GGML=0)."""
    return os.environ.get("SD_USE_VENDORED_GGML") == "0"


# --------------------------------------------------------------------------
# base classes
# --------------------------------------------------------------------------
class AbstractBuilder(ShellCmd):
    """Common builder scaffolding: paths + lib discovery/copy."""

    name: str = ""
    version: str = ""
    repo_url: str = ""

    def __init__(self, version: str | None = None, project: Project | None = None) -> None:
        proj = project or Project()
        super().__init__(dry_run=proj.dry_run)
        self.version = version or self.version
        self.project = proj

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.name}@{self.version}>"

    # paths ---------------------------------------------------------------
    @property
    def src_dir(self) -> Path:
        return self.project.src / self.name

    @property
    def build_dir(self) -> Path:
        return self.src_dir / "build"

    @property
    def prefix(self) -> Path:
        return self.project.install / self.name.lower()

    @property
    def include(self) -> Path:
        return self.prefix / "include"

    @property
    def lib(self) -> Path:
        return self.prefix / "lib"

    # lib discovery / copy ------------------------------------------------
    def get_lib_path(self, build_dir: Path, subdir: str, name: str) -> Path:
        """Platform-specific static-library path under a CMake build dir."""
        base = build_dir / subdir
        if PLATFORM == "Windows":
            for cfg in ("Release", "", "RelWithDebInfo", "MinSizeRel", "Debug"):
                p = (base / cfg / f"{name}.lib") if cfg else (base / f"{name}.lib")
                if p.exists():
                    return p
            return base / "Release" / f"{name}.lib"
        return base / f"lib{name}.a"

    def copy_lib(
        self, build_dir: Path, subdir: str, name: str, dest: Path, required: bool = True
    ) -> bool:
        lib_path = self.get_lib_path(build_dir, subdir, name)
        if self.dry_run or lib_path.exists():
            self.copy(lib_path, dest)
            return True
        if required:
            raise FileNotFoundError(f"Required library not found: {lib_path}")
        self.log.warning("Optional library not found: %s", lib_path)
        return False


class Builder(AbstractBuilder):
    """Concrete builder: shallow-clones repo_url at self.version (idempotent)."""

    def setup(self) -> None:
        self.project.setup()
        if self.src_dir.exists():
            self.log.info("%s already checked out at %s", self.name, self.src_dir)
            return
        self.log.info("fetching %s %s", self.name, self.version or "(default branch)")
        self.git_clone(
            self.repo_url,
            branch=self.version or None,
            recurse=True,
            cwd=self.project.src,
        )


class GgmlBuilder(Builder):
    """Base for ggml-backed engines: maps GGML_*/SD_* env flags to CMake options."""

    CUDA_TUNING_ENV_FLAGS: tuple[str, ...] = (
        "GGML_CUDA_FORCE_MMQ",
        "GGML_CUDA_FORCE_CUBLAS",
        "GGML_CUDA_PEER_MAX_BATCH_SIZE",
        "GGML_CUDA_FA_ALL_QUANTS",
    )
    BACKEND_SHORT_NAMES: dict[str, str] = {
        "GGML_METAL": "metal",
        "GGML_CUDA": "cuda",
        "GGML_VULKAN": "vulkan",
        "GGML_SYCL": "sycl",
        "GGML_HIP": "hip",
        "GGML_OPENCL": "opencl",
    }

    def get_backend_cmake_options(self) -> dict[str, Any]:
        raise NotImplementedError

    # shared helpers ------------------------------------------------------
    def enabled_backends_from_env(self) -> list[str]:
        result = []
        for env_name, short in self.BACKEND_SHORT_NAMES.items():
            default = env_name == "GGML_METAL" and PLATFORM == "Darwin"
            if getenv(env_name, default=default):
                result.append(short)
        return result

    def _set_backend(
        self, options: dict, cmake_name: str, enabled: bool, label: str, suffix: str = ""
    ) -> None:
        options[cmake_name] = "ON" if enabled else "OFF"
        if enabled:
            self.log.info("Enabling %s backend%s", label, suffix)

    def _apply_cuda_extras(self, options: dict) -> None:
        for env, key in (
            ("CMAKE_CUDA_ARCHITECTURES", "CMAKE_CUDA_ARCHITECTURES"),
            ("CMAKE_CUDA_COMPILER", "CMAKE_CUDA_COMPILER"),
        ):
            val = os.environ.get(env)
            if val:
                options[key] = val
        forward_env_flags(options, self.CUDA_TUNING_ENV_FLAGS)

    def _apply_hip_archs(self, options: dict) -> None:
        archs = os.environ.get("CMAKE_HIP_ARCHITECTURES")
        if archs:
            options["CMAKE_HIP_ARCHITECTURES"] = archs

    def _apply_openmp(self, options: dict) -> None:
        openmp = os.environ.get("GGML_OPENMP")
        if openmp is not None:
            options["GGML_OPENMP"] = "ON" if openmp == "1" else "OFF"

    def _apply_ggml_native(self, options: dict) -> None:
        native = os.environ.get("GGML_NATIVE")
        if native is not None:
            options["GGML_NATIVE"] = "ON" if native == "1" else "OFF"
        if getenv("GGML_CPU_ALL_VARIANTS", default=False):
            options["GGML_CPU_ALL_VARIANTS"] = "ON"
            options["GGML_NATIVE"] = "OFF"

    def _ggml_backend_options(self, prefix: str, suffix: str = "") -> dict[str, Any]:
        """Build the backend ON/OFF map for a GGML_*-style or SD_*-style prefix."""
        options: dict[str, Any] = {}
        metal = bool(getenv("GGML_METAL", default=(PLATFORM == "Darwin"))) and (
            PLATFORM == "Darwin"
        )
        cuda = bool(getenv("GGML_CUDA", default=False))
        vulkan = bool(getenv("GGML_VULKAN", default=False))
        sycl = bool(getenv("GGML_SYCL", default=False))
        hip = bool(getenv("GGML_HIP", default=False))
        opencl = bool(getenv("GGML_OPENCL", default=False))

        hip_flag = "SD_HIPBLAS" if prefix == "SD" else f"{prefix}_HIP"
        self._set_backend(options, f"{prefix}_METAL", metal, "Metal", suffix)
        self._set_backend(options, f"{prefix}_CUDA", cuda, "CUDA", suffix)
        if cuda:
            self._apply_cuda_extras(options)
        self._set_backend(options, f"{prefix}_VULKAN", vulkan, "Vulkan", suffix)
        self._set_backend(options, f"{prefix}_SYCL", sycl, "SYCL", suffix)
        self._set_backend(options, hip_flag, hip, "HIP/ROCm", suffix)
        if hip:
            self._apply_hip_archs(options)
        self._set_backend(options, f"{prefix}_OPENCL", opencl, "OpenCL", suffix)
        return options


# --------------------------------------------------------------------------
# llama.cpp
# --------------------------------------------------------------------------
class LlamaCppBuilder(GgmlBuilder):
    """Builds llama.cpp -- the engine that provides the ONE shared ggml."""

    name = "llama.cpp"
    version = LLAMACPP_VERSION
    repo_url = "https://github.com/ggml-org/llama.cpp.git"

    def __init__(
        self,
        version: str | None = None,
        project: Project | None = None,
        with_server: bool = False,
        openssl: bool = False,
    ) -> None:
        super().__init__(version=version, project=project)
        # server-context + cpp-httplib are only needed by consumers that embed
        # llama-server (chimera). cyllama/inferna leave them off.
        self.with_server = with_server
        self.openssl = openssl

    def get_backend_cmake_options(self) -> dict[str, Any]:
        options = self._ggml_backend_options("GGML")
        if getenv("GGML_BLAS", default=False):
            options["GGML_BLAS"] = "ON"
            vendor = os.environ.get("GGML_BLAS_VENDOR")
            if vendor:
                options["GGML_BLAS_VENDOR"] = vendor
        self._apply_openmp(options)
        self._apply_ggml_native(options)
        return options

    def _targets(self) -> list[str]:
        targets = ["llama", "llama-common", "mtmd"]
        if self.with_server:
            targets += ["server-context", "cpp-httplib"]
        return targets

    def _copy_headers(self) -> None:
        self.glob_copy(self.src_dir / "common", self.include, patterns=["*.h", "*.hpp"])
        self.glob_copy(self.src_dir / "ggml" / "include", self.include, patterns=["*.h"])
        self.glob_copy(self.src_dir / "include", self.include, patterns=["*.h"])
        # jinja (required by chat.h)
        if (self.src_dir / "common" / "jinja").exists():
            self.glob_copy(
                self.src_dir / "common" / "jinja",
                self.include / "jinja",
                patterns=["*.h", "*.hpp"],
            )
        # nlohmann json (required by json-partial.h)
        if (self.src_dir / "vendor" / "nlohmann").exists():
            self.glob_copy(
                self.src_dir / "vendor" / "nlohmann",
                self.include / "nlohmann",
                patterns=["*.hpp"],
            )
        # mtmd (multimodal)
        self.glob_copy(self.src_dir / "tools" / "mtmd", self.include, patterns=["*.h"])
        if self.with_server:
            self.glob_copy(
                self.src_dir / "tools" / "server", self.include, patterns=["server-*.h"]
            )
            httplib = self.src_dir / "vendor" / "cpp-httplib"
            if httplib.exists():
                self.glob_copy(httplib, self.include / "cpp-httplib", patterns=["httplib.h"])

    def _copy_libs(self) -> None:
        if not self.dry_run:
            self.lib.mkdir(parents=True, exist_ok=True)
        self.copy_lib(self.build_dir, "common", "llama-common", self.lib)
        self.copy_lib(self.build_dir, "src", "llama", self.lib)
        self.copy_lib(self.build_dir, "ggml/src", "ggml", self.lib)
        self.copy_lib(self.build_dir, "ggml/src", "ggml-base", self.lib)
        self.copy_lib(self.build_dir, "ggml/src", "ggml-cpu", self.lib)
        self.copy_lib(self.build_dir, "tools/mtmd", "mtmd", self.lib)
        if self.with_server:
            self.copy_lib(self.build_dir, "tools/server", "server-context", self.lib)
            self.copy_lib(
                self.build_dir, "vendor/cpp-httplib", "cpp-httplib", self.lib, required=False
            )
        self._copy_backend_libs()

    def _copy_backend_libs(self) -> None:
        enabled = self.enabled_backends_from_env()
        if "metal" in enabled:
            self.copy_lib(
                self.build_dir, "ggml/src/ggml-blas", "ggml-blas", self.lib, required=False
            )
        for short in enabled:
            self.copy_lib(
                self.build_dir, f"ggml/src/ggml-{short}", f"ggml-{short}", self.lib,
                required=False,
            )

    def build(self) -> None:
        self.setup()
        self.log.info("building %s (with_server=%s)", self.name, self.with_server)
        backend_options = self.get_backend_cmake_options()

        # When SD will share this ggml, match its required GGML_MAX_NAME so the
        # ggml_tensor struct layout is identical on both sides.
        extra: dict[str, Any] = {}
        if shared_ggml_enabled():
            _def = f"-DGGML_MAX_NAME={GGML_MAX_NAME}"
            extra["CMAKE_C_FLAGS"] = _def
            extra["CMAKE_CXX_FLAGS"] = _def

        self.cmake_config(
            self.src_dir,
            self.build_dir,
            BUILD_SHARED_LIBS=False,
            CMAKE_POSITION_INDEPENDENT_CODE=True,
            CMAKE_CXX_VISIBILITY_PRESET="hidden",
            CMAKE_C_VISIBILITY_PRESET="hidden",
            CMAKE_VISIBILITY_INLINES_HIDDEN=True,
            LLAMA_CURL=False,
            LLAMA_OPENSSL=self.openssl,
            LLAMA_BUILD_SERVER=self.with_server,
            LLAMA_BUILD_WEBUI=False,
            LLAMA_BUILD_TESTS=False,
            LLAMA_BUILD_EXAMPLES=False,
            **extra,
            **backend_options,
        )
        self.cmake_build_targets(self.build_dir, self._targets(), release=True)
        if not self.dry_run:
            self.include.mkdir(parents=True, exist_ok=True)
        self._copy_headers()
        self._copy_libs()


# --------------------------------------------------------------------------
# whisper.cpp
# --------------------------------------------------------------------------
class WhisperCppBuilder(GgmlBuilder):
    """Builds whisper.cpp; stages libwhisper.a + libcommon.a only.

    whisper builds its own ggml, which we do NOT stage -- the consumer links
    libwhisper.a against llama.cpp's shared ggml. (Matches the three projects;
    note this relies on whisper's pinned version being ABI-compatible with
    llama's ggml -- see README "ABI notes".)
    """

    name = "whisper.cpp"
    version = WHISPERCPP_VERSION
    repo_url = "https://github.com/ggml-org/whisper.cpp"

    def get_backend_cmake_options(self) -> dict[str, Any]:
        options = self._ggml_backend_options("GGML", suffix=" for whisper.cpp")
        self._apply_openmp(options)
        self._apply_ggml_native(options)
        return options

    def build(self) -> None:
        self.setup()
        self.log.info("building %s", self.name)
        backend_options = self.get_backend_cmake_options()
        self.cmake_config(
            self.src_dir,
            self.build_dir,
            BUILD_SHARED_LIBS=False,
            CMAKE_POSITION_INDEPENDENT_CODE=True,
            CMAKE_CXX_VISIBILITY_PRESET="hidden",
            CMAKE_C_VISIBILITY_PRESET="hidden",
            CMAKE_VISIBILITY_INLINES_HIDDEN=True,
            CMAKE_INSTALL_LIBDIR="lib",
            **backend_options,
        )
        self.cmake_build(self.build_dir, release=True)
        if not self.dry_run:
            self.include.mkdir(parents=True, exist_ok=True)
        # whisper.h lives in include/; common.h under examples/.
        self.glob_copy(self.src_dir / "include", self.include, patterns=["*.h"])
        if self.dry_run or (self.src_dir / "examples").exists():
            self.glob_copy(
                self.src_dir / "examples", self.include, patterns=["*.h", "*.hpp"]
            )
        if not self.dry_run:
            self.lib.mkdir(parents=True, exist_ok=True)
        self.copy_lib(self.build_dir, "src", "whisper", self.lib, required=False)
        # Older layouts emit libwhisper.a at the build root.
        self.copy_lib(self.build_dir, ".", "whisper", self.lib, required=False)
        self.copy_lib(self.build_dir, "examples", "common", self.lib, required=False)


# --------------------------------------------------------------------------
# stable-diffusion.cpp
# --------------------------------------------------------------------------
class StableDiffusionCppBuilder(GgmlBuilder):
    """Builds stable-diffusion.cpp, optionally against llama.cpp's shared ggml.

    The single-ggml mechanism: when SD_USE_VENDORED_GGML=0, replace SD's
    vendored ggml/ subtree with llama.cpp's (so enum/struct ABI matches the
    ggml we link at runtime) before compiling.
    """

    name = "stable-diffusion.cpp"
    version = SDCPP_VERSION
    repo_url = "https://github.com/leejet/stable-diffusion.cpp.git"

    def get_backend_cmake_options(self) -> dict[str, Any]:
        options = self._ggml_backend_options("SD", suffix=" for stable-diffusion.cpp")
        self._apply_openmp(options)
        return options

    def _sync_ggml_abi(self) -> None:
        """Replace SD's vendored ggml with llama.cpp's so enum/struct ids agree."""
        llama_ggml = self.project.src / "llama.cpp" / "ggml"
        sd_ggml = self.src_dir / "ggml"
        self.log.info("syncing SD ggml from llama.cpp (%s -> %s)", llama_ggml, sd_ggml)
        if self.dry_run:
            return
        if not llama_ggml.exists() or not sd_ggml.exists():
            self.log.warning("Cannot sync ggml ABI: llama.cpp or SD ggml dir missing")
            return
        shutil.rmtree(sd_ggml)
        shutil.copytree(llama_ggml, sd_ggml)

    def build(self, examples: bool = False) -> None:
        self.setup()
        self.log.info("building %s (shared_ggml=%s)", self.name, shared_ggml_enabled())
        if shared_ggml_enabled():
            self._sync_ggml_abi()
        backend_options = self.get_backend_cmake_options()
        self.cmake_config(
            self.src_dir,
            self.build_dir,
            BUILD_SHARED_LIBS=False,
            CMAKE_POSITION_INDEPENDENT_CODE=True,
            CMAKE_CXX_VISIBILITY_PRESET="hidden",
            CMAKE_C_VISIBILITY_PRESET="hidden",
            CMAKE_VISIBILITY_INLINES_HIDDEN=True,
            CMAKE_INSTALL_LIBDIR="lib",
            SD_BUILD_EXAMPLES=examples,
            SD_WEBP=getenv("SD_WEBP", default=False),
            SD_WEBM=getenv("SD_WEBM", default=False),
            **backend_options,
        )
        self.cmake_build(self.build_dir, release=True)
        if not self.dry_run:
            self.include.mkdir(parents=True, exist_ok=True)
        self.glob_copy(self.src_dir, self.include, patterns=["*.h", "*.hpp"])
        stb = self.src_dir / "thirdparty"
        if stb.exists():
            for h in ("stb_image.h", "stb_image_write.h", "stb_image_resize.h"):
                if (stb / h).exists():
                    self.copy(stb / h, self.include)
        if not self.dry_run:
            self.lib.mkdir(parents=True, exist_ok=True)
        self.copy_lib(self.build_dir, ".", "stable-diffusion", self.lib)
