# UGTS-DTI Makefile
# Inspired by pvrptw setup

.PHONY: help install lint format check clean verify

# Default target
help:
	@echo "UGTS-DTI Development Commands:"
	@echo ""
	@echo "  install       Install dependencies (uv sync --dev)"
	@echo "  lint          Run linting and type checking (ruff + mypy)"
	@echo "  format        Format code with ruff"
	@echo "  check         Run all quality checks"
	@echo "  verify        Full environment verification"
	@echo "  clean         Clean cache and build artifacts"
	@echo ""
	@echo "Usage:"
	@echo "  make install"
	@echo "  make check"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	uv sync --dev

# Run linting and type checking
lint:
	@echo "🔍 Running code quality checks..."
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src

# Format code
format:
	@echo "📝 Formatting code..."
	uv run ruff format .
	uv run ruff check --fix .

# Run all checks
check: lint
	@echo "✅ All quality checks passed!"

# Full environment verification
verify: install check
	@echo "✅ Full verification complete - environment ready for research!"

# Clean cache and artifacts
clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .ruff_cache
	rm -rf .mypy_cache
	uv cache clean
