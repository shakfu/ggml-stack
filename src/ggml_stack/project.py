"""Directory layout for a single-ggml build.

The three source projects all use the same shape:

    <root>/build/<engine>/          upstream source checkout (+ /build for CMake)
    <root>/thirdparty/<engine>/     staged output: include/ lib/ [src-aux/]

`Project` parameterizes the two roots so a consumer can point them wherever
its own `manage.py` expects (chimera/cyllama/inferna all use build/ and
thirdparty/ under the repo root, which are the defaults here).
"""

from __future__ import annotations

from pathlib import Path

from .shell import ShellCmd


class Project(ShellCmd):
    """Holds the build/ (sources) and thirdparty/ (staged libs) roots."""

    def __init__(
        self,
        root: Path | str | None = None,
        build_dirname: str = "build",
        thirdparty_dirname: str = "thirdparty",
        dry_run: bool = False,
    ) -> None:
        super().__init__(dry_run=dry_run)
        self.cwd = Path(root).resolve() if root else Path.cwd()
        # Source checkouts land under build/ (matches all three projects, where
        # Project.src == Project.build).
        self.build = self.cwd / build_dirname
        self.src = self.build
        # Built libraries/headers are staged under thirdparty/.
        self.thirdparty = self.cwd / thirdparty_dirname
        self.install = self.thirdparty

    def setup(self) -> None:
        if self.dry_run:
            return
        self.build.mkdir(parents=True, exist_ok=True)
        self.install.mkdir(parents=True, exist_ok=True)

    def __repr__(self) -> str:
        return f"<Project root={self.cwd} build={self.build} thirdparty={self.thirdparty}>"
