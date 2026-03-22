.PHONY: test cov cov-html lint format check

# Run tests (installs/uses the optional 'test' extra via uv)
test:
	uv run --extra test pytest -q

# Run tests with coverage (adjust --cov=... to your package if needed)

cov:
	uv run --extra test pytest --cov=iartisanz --cov-report=term-missing

# Optional: HTML coverage report in htmlcov/

cov-html:
	uv run --extra test pytest --cov=iartisanz --cov-report=html

# Lint: check for code quality issues
lint:
	uv run --extra dev ruff check src/

# Format: auto-format code
format:
	uv run --extra dev ruff format src/
	uv run --extra dev ruff check --fix src/

# Check: lint + format check (CI-friendly, no modifications)
check:
	uv run --extra dev ruff format --check src/
	uv run --extra dev ruff check src/
