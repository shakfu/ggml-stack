"""Dry-run tests for the single-ggml recipe.

These assert on the *generated* git/cmake command lines (via dry-run, no
compilation), so they validate the load-bearing wiring -- build order, the
shared-ggml flags, GGML_MAX_NAME propagation, the SD ggml sync -- fast and on
any machine without a toolchain.
"""

from __future__ import annotations

import os

import pytest

from ggml_stack import (
    GGML_MAX_NAME,
    LlamaCppBuilder,
    Project,
    StableDiffusionCppBuilder,
    WhisperCppBuilder,
    shared_ggml_enabled,
)
from ggml_stack.shell import ShellCmd


# --------------------------------------------------------------------------
# cmake flag formatting
# --------------------------------------------------------------------------
def test_cmake_flag_bool_and_semicolon():
    f = ShellCmd._cmake_flag
    assert f("LLAMA_CURL", False) == "-DLLAMA_CURL=OFF"
    assert f("LLAMA_BUILD_SERVER", True) == "-DLLAMA_BUILD_SERVER=ON"
    assert f("CMAKE_CUDA_ARCHITECTURES", "75;86") == '-DCMAKE_CUDA_ARCHITECTURES="75;86"'
    assert f("X", 3) == "-DX=3"


# --------------------------------------------------------------------------
# shared-ggml flag
# --------------------------------------------------------------------------
def test_shared_ggml_enabled(monkeypatch):
    monkeypatch.setenv("SD_USE_VENDORED_GGML", "0")
    assert shared_ggml_enabled() is True
    monkeypatch.setenv("SD_USE_VENDORED_GGML", "1")
    assert shared_ggml_enabled() is False
    monkeypatch.delenv("SD_USE_VENDORED_GGML", raising=False)
    assert shared_ggml_enabled() is False


# --------------------------------------------------------------------------
# backend option mapping
# --------------------------------------------------------------------------
def test_llama_backend_metal_default_on_darwin(monkeypatch):
    monkeypatch.setattr("ggml_stack.builders.PLATFORM", "Darwin")
    monkeypatch.delenv("GGML_METAL", raising=False)
    opts = LlamaCppBuilder(project=Project(dry_run=True)).get_backend_cmake_options()
    assert opts["GGML_METAL"] == "ON"
    assert opts["GGML_CUDA"] == "OFF"


def test_llama_backend_cuda_off_by_default(monkeypatch):
    monkeypatch.setattr("ggml_stack.builders.PLATFORM", "Linux")
    for v in ("GGML_METAL", "GGML_CUDA", "GGML_VULKAN", "GGML_HIP"):
        monkeypatch.delenv(v, raising=False)
    opts = LlamaCppBuilder(project=Project(dry_run=True)).get_backend_cmake_options()
    assert opts["GGML_METAL"] == "OFF"
    assert opts["GGML_CUDA"] == "OFF"


def test_ggml_native_off_propagates(monkeypatch):
    monkeypatch.setenv("GGML_NATIVE", "0")
    opts = LlamaCppBuilder(project=Project(dry_run=True)).get_backend_cmake_options()
    assert opts["GGML_NATIVE"] == "OFF"


def test_sd_uses_sd_prefixed_backend_flags(monkeypatch):
    monkeypatch.setattr("ggml_stack.builders.PLATFORM", "Linux")
    monkeypatch.setenv("GGML_CUDA", "1")
    opts = StableDiffusionCppBuilder(project=Project(dry_run=True)).get_backend_cmake_options()
    assert opts["SD_CUDA"] == "ON"
    assert "GGML_CUDA" not in opts  # SD uses SD_* names


# --------------------------------------------------------------------------
# single-ggml wiring through a full dry-run build
# --------------------------------------------------------------------------
def _llama_configure_cmd(monkeypatch, shared: bool):
    if shared:
        monkeypatch.setenv("SD_USE_VENDORED_GGML", "0")
    else:
        monkeypatch.delenv("SD_USE_VENDORED_GGML", raising=False)
    b = LlamaCppBuilder(project=Project(dry_run=True))
    b.build()
    return [c for c in b.commands if c.startswith("cmake -S")][0]


def test_llama_max_name_only_when_shared(monkeypatch):
    shared = _llama_configure_cmd(monkeypatch, shared=True)
    assert f"-DCMAKE_C_FLAGS=-DGGML_MAX_NAME={GGML_MAX_NAME}" in shared
    assert f"-DCMAKE_CXX_FLAGS=-DGGML_MAX_NAME={GGML_MAX_NAME}" in shared

    unshared = _llama_configure_cmd(monkeypatch, shared=False)
    assert "GGML_MAX_NAME" not in unshared


def test_llama_server_flags(monkeypatch):
    monkeypatch.delenv("SD_USE_VENDORED_GGML", raising=False)
    b = LlamaCppBuilder(project=Project(dry_run=True), with_server=True, openssl=True)
    b.build()
    cfg = [c for c in b.commands if c.startswith("cmake -S")][0]
    assert "-DLLAMA_BUILD_SERVER=ON" in cfg
    assert "-DLLAMA_OPENSSL=ON" in cfg
    targets = [c for c in b.commands if "--target" in c][0]
    assert "--target server-context" in targets and "--target cpp-httplib" in targets

    b2 = LlamaCppBuilder(project=Project(dry_run=True))  # defaults: no server
    b2.build()
    cfg2 = [c for c in b2.commands if c.startswith("cmake -S")][0]
    assert "-DLLAMA_BUILD_SERVER=OFF" in cfg2
    assert "-DLLAMA_OPENSSL=OFF" in cfg2


def test_sd_syncs_ggml_only_when_shared(monkeypatch, caplog):
    import logging

    monkeypatch.setenv("SD_USE_VENDORED_GGML", "0")
    with caplog.at_level(logging.INFO):
        StableDiffusionCppBuilder(project=Project(dry_run=True)).build()
    assert any("syncing SD ggml from llama.cpp" in r.message for r in caplog.records)

    caplog.clear()
    monkeypatch.delenv("SD_USE_VENDORED_GGML", raising=False)
    with caplog.at_level(logging.INFO):
        StableDiffusionCppBuilder(project=Project(dry_run=True)).build()
    assert not any("syncing SD ggml" in r.message for r in caplog.records)


def test_build_stack_order_and_shared_env(monkeypatch):
    from ggml_stack import StackConfig, build_stack

    monkeypatch.delenv("SD_USE_VENDORED_GGML", raising=False)
    # Patch each builder.build to record invocation order instead of compiling.
    order: list[str] = []
    import ggml_stack.recipe as recipe

    for cls in ("LlamaCppBuilder", "WhisperCppBuilder", "StableDiffusionCppBuilder"):
        def make(name):
            def _build(self, *a, **k):
                order.append(name)
            return _build
        monkeypatch.setattr(getattr(recipe, cls), "build", make(cls))

    build_stack(StackConfig(dry_run=True))
    assert order == ["LlamaCppBuilder", "WhisperCppBuilder", "StableDiffusionCppBuilder"]
    # build_stack must have set the shared-ggml env for the builders.
    assert os.environ.get("SD_USE_VENDORED_GGML") == "0"


def test_shared_sd_requires_llama(monkeypatch):
    from ggml_stack import StackConfig, build_stack

    with pytest.raises(ValueError, match="requires building 'llama'"):
        build_stack(StackConfig(engines=("sd",), shared_ggml=True, dry_run=True))


def test_project_layout(tmp_path):
    p = Project(root=tmp_path)
    assert p.build == tmp_path / "build"
    assert p.thirdparty == tmp_path / "thirdparty"
    b = LlamaCppBuilder(project=p)
    assert b.src_dir == tmp_path / "build" / "llama.cpp"
    assert b.lib == tmp_path / "thirdparty" / "llama.cpp" / "lib"
