.PHONY: test cov cov-html

# Run tests (installs/uses the optional 'test' extra via uv)
test:
	uv run --extra test pytest -q

# Run tests with coverage (adjust --cov=... to your package if needed)

cov:
	uv run --extra test pytest --cov=iartisanz --cov-report=term-missing

# Optional: HTML coverage report in htmlcov/

cov-html:
	uv run --extra test pytest --cov=iartisanz --cov-report=html
