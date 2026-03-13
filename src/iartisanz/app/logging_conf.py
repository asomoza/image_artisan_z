import os


def get_logging_config(debug=False):
    console_level = "DEBUG" if debug else "ERROR"

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "console_formatter": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
            "file_formatter": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
            },
        },
        "handlers": {
            "consoleHandler": {
                "class": "logging.StreamHandler",
                "level": console_level,
                "formatter": "console_formatter",
                "stream": "ext://sys.stdout",
            },
            "fileHandler": {
                "class": "logging.FileHandler",
                "level": "ERROR",
                "formatter": "file_formatter",
                "filename": os.path.join(os.path.expanduser("~"), ".iartisanxl", "iartisanxl.log"),
            },
        },
        "loggers": {
            "": {
                "level": "DEBUG",
                "handlers": ["consoleHandler", "fileHandler"],
            },
        },
    }


# Backward compat: keep the dict available for any imports that reference it directly
logging_config = get_logging_config(debug=False)
