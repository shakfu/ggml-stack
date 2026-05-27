# ggml-stack -- developer task runner
#
# This project is managed with `uv`. Targets wrap the common dev loop:
# environment sync, tests, lint/format/typecheck, the CLI, and packaging.

UV ?= uv
PKG := ggml_stack
SRC := src
TESTS := tests

.DEFAULT_GOAL := build

.PHONY: help install lock test lint format format-check typecheck check \
        build run info dry-run dist clean distclean reset

## Show this help
help:
	@awk '/^## / { sub(/^## /, ""); desc = $$0; next } \
	      /^[a-zA-Z0-9_-]+:/ { name = $$1; sub(/:.*/, "", name); \
	                           if (desc != "") { printf "  \033[36m%-14s\033[0m %s\n", name, desc; desc = "" } next } \
	      { desc = "" }' $(MAKEFILE_LIST)

# --- environment ----------------------------------------------------------

## Create/refresh the virtualenv with test extras (uv sync)
install:
	@uv sync

## Re-resolve and update uv.lock
lock:
	@uv lock

# --- quality --------------------------------------------------------------

## Run the test suite
test:
	@uv run pytest

## Lint with ruff (no install needed via uvx)
lint:
	@uvx ruff check $(SRC) $(TESTS)

## Auto-format with ruff
format:
	@uvx ruff format $(SRC) $(TESTS)

## Check formatting without writing changes
format-check:
	@uvx ruff format --check $(SRC) $(TESTS)

## Static type-check with mypy
typecheck:
	@uvx mypy $(SRC)

## Run lint, format-check, and tests
check: lint format-check test

# --- CLI ------------------------------------------------------------------

## Fetch + build the engines; pass flags via ARGS="--engines llama --with-server"
build: install
	@uv run ggml-stack build $(ARGS)

## Run the CLI; pass args via ARGS="build --dry-run"
run:
	@uv run ggml-stack $(ARGS)

## Show staged archives under thirdparty/
info:
	@uv run ggml-stack info

## Print the git/cmake command lines without executing
dry-run:
	@uv run ggml-stack build --dry-run

# --- packaging ------------------------------------------------------------

## Build wheel and sdist into dist/
dist:
	@uv build

# --- housekeeping ---------------------------------------------------------

## Remove build artifacts and caches
clean:
	@rm -rf build dist wheels *.egg-info src/*.egg-info
	@find . -type d -name __pycache__ -prune -exec rm -rf {} +
	@find . -type d -name '.*_cache' -prune -exec rm -rf {} +

## Also remove the virtualenv
distclean: clean
	@rm -rf .venv


## Remove everythin
reset: distclean
