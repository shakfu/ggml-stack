"""Shell / filesystem / CMake helpers shared by the engine builders.

This is the generic command layer extracted (verbatim in behavior) from the
`ShellCmd` base classes in the chimera / cyllama / inferna `manage.py` files.
Nothing here is engine-specific.
"""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Iterable, Union
from urllib.request import urlretrieve

Pathlike = Union[str, Path]

PLATFORM = platform.system()
ARCH = platform.machine()


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------
class CustomFormatter(logging.Formatter):
    """Colorized, compact log formatter (matches the three manage.py files)."""

    grey = "\x1b[38;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    fmt = "%(asctime)s - {}%(levelname)s{} - %(name)s.%(funcName)s - %(message)s"

    FORMATS = {
        logging.DEBUG: fmt.format(grey, reset),
        logging.INFO: fmt.format(green, reset),
        logging.WARNING: fmt.format(yellow, reset),
        logging.ERROR: fmt.format(red, reset),
        logging.CRITICAL: fmt.format(bold_red, reset),
    }

    def format(self, record: logging.LogRecord) -> str:
        formatter = logging.Formatter(
            self.FORMATS.get(record.levelno), datefmt="%H:%M:%S"
        )
        return formatter.format(record)


def setup_logging(level: int = logging.INFO) -> None:
    """Install the colorized formatter on the root logger (idempotent)."""
    root = logging.getLogger()
    if any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.setLevel(level)
        return
    handler = logging.StreamHandler()
    handler.setFormatter(CustomFormatter())
    root.addHandler(handler)
    root.setLevel(level)


# --------------------------------------------------------------------------
# env helpers
# --------------------------------------------------------------------------
_TRUTHY = {"1", "true", "on", "yes"}
_FALSY = {"0", "false", "off", "no"}


def getenv(key: str, default: Union[bool, str, int] = False) -> Union[bool, str, int]:
    """Read an env var, coercing to the type of `default`.

    For a bool default, "1"/"true"/"on"/"yes" -> True, the falsy set -> False.
    Mirrors the `getenv` helper used in the source manage.py files.
    """
    raw = os.environ.get(key)
    if raw is None:
        return default
    if isinstance(default, bool):
        low = raw.strip().lower()
        if low in _TRUTHY:
            return True
        if low in _FALSY:
            return False
        return bool(raw)
    if isinstance(default, int):
        try:
            return int(raw)
        except ValueError:
            return default
    return raw


def setenv(key: str, default: str) -> str:
    """Set `key` to `default` if unset; return the effective value."""
    return os.environ.setdefault(key, default)


# --------------------------------------------------------------------------
# command runner
# --------------------------------------------------------------------------
class ShellCmd:
    """Thin wrapper over subprocess + filesystem ops with a logger.

    Set `dry_run=True` to log commands without executing them -- used by the
    test-suite to assert on generated CMake command lines without compiling.
    """

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.log = logging.getLogger(self.__class__.__name__)
        self.commands: list[str] = []  # recorded command history (handy for tests)

    # -- process ----------------------------------------------------------
    def cmd(self, shellcmd: str, cwd: Pathlike = ".") -> None:
        """Run a shell command within `cwd` (or just record it in dry-run)."""
        cwd_path = Path(cwd).resolve()
        self.log.info(shellcmd)
        self.commands.append(shellcmd)
        if self.dry_run:
            return
        try:
            subprocess.check_call(shellcmd, shell=True, cwd=str(cwd_path))
        except subprocess.CalledProcessError:
            self.log.critical("command failed: %s", shellcmd, exc_info=True)
            sys.exit(1)

    def git_clone(
        self,
        url: str,
        branch: str | None = None,
        directory: Pathlike | None = None,
        recurse: bool = False,
        cwd: Pathlike = ".",
    ) -> None:
        """Shallow `git clone` (optionally at a branch/tag, with submodules)."""
        _cmds = ["git clone --depth 1"]
        if branch:
            _cmds.append(f"--branch {branch}")
        if recurse:
            _cmds.append("--recurse-submodules --shallow-submodules")
        _cmds.append(url)
        if directory:
            _cmds.append(str(directory))
        self.cmd(" ".join(_cmds), cwd=cwd)

    # -- filesystem -------------------------------------------------------
    def copy(self, src: Pathlike, dst: Pathlike) -> None:
        """Copy a file or directory tree (like `cp -rf`)."""
        self.log.info("copy %s to %s", src, dst)
        if self.dry_run:
            return
        src, dst = Path(src), Path(dst)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)

    def glob_copy(self, src: Pathlike, dest: Pathlike, patterns: list[str]) -> None:
        """Copy files matching glob `patterns` from `src` to `dest`."""
        src, dest = Path(src), Path(dest)
        if not src.exists():
            if self.dry_run:
                self.log.info("glob_copy %s -> %s (skipped: no checkout in dry-run)", src, dest)
                return
            raise IOError(f"src dir '{src}' not found")
        if not self.dry_run:
            dest.mkdir(parents=True, exist_ok=True)
        for p in patterns:
            for f in src.glob(p):
                self.copy(f, dest)

    def download(
        self,
        url: str,
        tofolder: Pathlike | None = None,
        max_size: int = 1024 * 1024 * 100,
    ) -> Path:
        """Download `url` into `tofolder` with a size cap and basic validation."""
        if not url.startswith(("https://", "http://")):
            raise ValueError(f"Unsupported URL scheme: {url}")
        basename = os.path.basename(url)
        if ".." in basename or basename.startswith("/"):
            raise ValueError(f"Invalid filename in URL: {url}")
        _path = Path(basename)
        if tofolder:
            _path = Path(tofolder).resolve() / _path
            if _path.exists():
                return _path
        self.log.info("Downloading %s to %s", url, _path)
        if self.dry_run:
            return _path
        filename, _ = urlretrieve(url, filename=_path)
        if _path.stat().st_size > max_size:
            _path.unlink()
            raise ValueError(
                f"Downloaded file exceeds size limit: {_path.stat().st_size} > {max_size}"
            )
        return Path(filename)

    def extract(self, archive: Pathlike, tofolder: Pathlike = ".") -> None:
        """Extract a tar/zip archive with path-traversal protection."""
        if self.dry_run:
            return
        dest = Path(tofolder).resolve()
        if tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tar:
                for member in tar.getmembers():
                    if not str((dest / member.name).resolve()).startswith(str(dest)):
                        raise ValueError(f"path traversal in tar: {member.name}")
                tar.extractall(dest)
        elif zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    if not str((dest / info.filename).resolve()).startswith(str(dest)):
                        raise ValueError(f"path traversal in zip: {info.filename}")
                zf.extractall(dest)
        else:
            raise TypeError(f"cannot extract from {archive}")

    # -- cmake ------------------------------------------------------------
    @staticmethod
    def _cmake_flag(k: str, v: Union[str, bool, int]) -> str:
        if isinstance(v, bool):
            val: Union[str, int] = "ON" if v else "OFF"
        else:
            val = v
        if isinstance(val, str) and ";" in val:
            return f'-D{k}="{val}"'
        return f"-D{k}={val}"

    def cmake_config(
        self,
        src_dir: Pathlike,
        build_dir: Pathlike,
        *scripts: str,
        **options: Union[str, bool, int],
    ) -> None:
        """CMake configure/generate stage. Bools map to ON/OFF."""
        src_dir, build_dir = Path(src_dir), Path(build_dir)
        if not self.dry_run:
            if not src_dir.exists():
                raise FileNotFoundError(f"CMake source directory not found: {src_dir}")
            build_dir.mkdir(parents=True, exist_ok=True)
        _cmds = [f"cmake -S {src_dir} -B {build_dir}"]
        if scripts:
            _cmds.append(" ".join(f"-C {p}" for p in scripts))
        if options:
            _cmds.append(" ".join(self._cmake_flag(k, v) for k, v in options.items()))
        self.cmd(" ".join(_cmds))

    def cmake_build(self, build_dir: Pathlike, release: bool = False) -> None:
        _cmd = f"cmake --build {build_dir}"
        if release:
            _cmd += " --config Release"
        _cmd += f" --parallel {os.cpu_count() or 4}"
        self.cmd(_cmd)

    def cmake_build_targets(
        self, build_dir: Pathlike, targets: list[str], release: bool = False
    ) -> None:
        _cmd = f"cmake --build {build_dir}"
        if release:
            _cmd += " --config Release"
        for target in targets:
            _cmd += f" --target {target}"
        _cmd += f" --parallel {os.cpu_count() or 4}"
        self.cmd(_cmd)

    def cmake_install(self, build_dir: Pathlike, prefix: Pathlike | None = None) -> None:
        _cmds = ["cmake --install", str(build_dir)]
        if prefix:
            _cmds.append(f"--prefix {prefix}")
        self.cmd(" ".join(_cmds))


def forward_env_flags(options: dict, names: Iterable[str]) -> None:
    """Copy any set env vars in `names` straight into a CMake options dict."""
    for name in names:
        val = os.environ.get(name)
        if val is not None:
            options[name] = val
