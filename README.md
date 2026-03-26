# Image Artisan Z

Image Artisan Z is a desktop app for generating images with local models.
It is made to run on your computer without needing a browser. 
For the moment it just runs Z-Image models and Flux.2 models

This project is under active development.

## Requirements

- Python 3.12+
- An NVIDIA GPU is recommended for reasonable speed, but it can run on CPU (slow)

## Install (uv only)

1. Install `uv`: https://docs.astral.sh/uv/
2. From the repo folder, install dependencies:

```bash
uv sync
```

## Run

```bash
uv run iartisanz
```

On first start you will be asked to choose folders for models and outputs.

## Tests

Run the regular test suite:

```bash
uv run --extra test pytest -q
```

Or, using the Makefile:

```bash
make test
```

Run the heavier Hugging Face tests (downloads a small test model and may take longer):

```bash
IARTISANZ_RUN_HF_TESTS=1 uv run --extra test pytest -q -s -m hf
```

Skip Hugging Face tests:

```bash
uv run --extra test pytest -q -m "not hf"
```

## License

Apache License 2.0. See [LICENSE](LICENSE).