# Development Setup

This document explains how to set up the development environment for the ChunkFormer project.

## Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality and consistency.
The following tools are configured and will run automatically with pre-commit:

- **Black**: Code formatting with 100 character line length
- **isort**: Import sorting with Black-compatible profile
- **flake8**: Linting with max line length of 100
- **mypy**: Type checking (optional, continues on error)

To set up pre-commit hooks:

1. Install the development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Install the pre-commit hooks:
   ```bash
   pre-commit install
   ```

3. (Optional) Run pre-commit on all files:
   ```bash
   pre-commit run --all-files
   ```

## Running Tests

### Quick Commands
```bash
# Run all tests
pytest

# Verbose output
pytest -v -s
```


## Manual Commands

You can also run the tools manually:

```bash
# Format code
black --line-length=100 .

# Sort imports
isort --profile black --line-length=100 .

# Lint code (uses config from pyproject.toml)
flake8 .

# Or with explicit parameters
flake8 . --max-line-length=100 --extend-ignore=E203,W503,B008,C416,EXE001

# Type check
mypy chunkformer --ignore-missing-imports

# Run all pre-commit hooks
pre-commit run --all-files
```
