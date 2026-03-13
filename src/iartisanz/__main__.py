import argparse
import logging
import logging.config
import os
import sys
import warnings

from iartisanz.app.iartisanz_application import BaseTorchAppApplication
from iartisanz.app.logging_conf import get_logging_config


logging.getLogger("PIL").setLevel(logging.ERROR)


def my_exception_hook(exctype, value, traceback):
    print(f"Unhandled exception: {value}")
    sys.__excepthook__(exctype, value, traceback)


sys.excepthook = my_exception_hook


def main():
    parser = argparse.ArgumentParser(description="Image Artisan Z")
    parser.add_argument("--debug", action="store_true", help="Show all log messages (default: errors only)")
    args = parser.parse_args()

    logging_config = get_logging_config(debug=args.debug)
    os.makedirs(
        os.path.dirname(logging_config["handlers"]["fileHandler"]["filename"]),
        exist_ok=True,
    )
    logging.config.dictConfig(logging_config)

    if not args.debug:
        warnings.filterwarnings("ignore", message=".*Attention backends are an experimental feature.*")

    from iartisanz.app.model_manager import ModelManager, set_global_model_manager

    set_global_model_manager(ModelManager())
    app = BaseTorchAppApplication(sys.argv)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
