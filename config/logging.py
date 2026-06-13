import logging
import logging.config
from pathlib import Path
from typing import Optional
from pythonjsonlogger import jsonlogger

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "app.log" 
ERROR_LOG_FILE = LOG_DIR / "error.log"

NOISY_LOGGERS = {
    "httpx", "httpcore", "uvicorn.access",
    "langchain", "openai", "qdrant_client",
    "sentence_transformers", "transformers"
}

LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class PipelineJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON log formatter.

    Adds extra information like:
    - module name, function name, and line number
    - thread information

    Makes logs consistent and easier to use in logging systems
    """

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["module"] = record.module
        log_record["func"] = record.funcName
        log_record["line"] = record.lineno
        log_record["thread"] = record.thread
        log_record["thread_name"] = record.threadName


def setup_logging(log_level="INFO", json_logs=False, log_to_file=True):
    """
    Configure global logging for the application.

    This function sets up:
    - Console logging (human-readable or JSON)
    - Rotating file logging (app.log)
    - Error-only logging (error.log)
    - Logging levels per module (provided as parameter)
    - Suppression of noisy third-party libraries

    Args:
        log_level (str): Global logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_logs (bool): If True, logs are output in JSON format (recommended for production)
        log_to_file (bool): If True, logs are written to rotating log files
    """
     
    level = _validate(log_level)
    LOG_DIR.mkdir(exist_ok=True)

    handlers = {
        "console": {
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
            "formatter": "json" if json_logs else "console",
        }
    }

    if log_to_file:
        handlers.update({
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(LOG_FILE),
                "maxBytes": 10_000_000,
                "backupCount": 2,
                "formatter": "json",
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(ERROR_LOG_FILE),
                "maxBytes": 5_000_000,
                "backupCount": 2,
                "level": "ERROR",
                "formatter": "json",
            },
        })

    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,

        "formatters": {
            "console": {
                "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            },
            "json": {
                "()": PipelineJsonFormatter,
            },
        },

        "handlers": handlers,

        "root": {
            "level": level,
            "handlers": list(handlers.keys()),
        },

        "loggers": {
            **{name: {"level": "WARNING"} for name in NOISY_LOGGERS},
            "modules": {"level": level},
            "rag": {"level": level},
            "api": {"level": level},
            "config": {"level": level},
        },
    })

    logging.getLogger(__name__).info("Logging initialized", extra={
        "level": level,
        "json_logs": json_logs,
        "log_to_file": log_to_file,
    })


def get_logger(name: str):
    return logging.getLogger(name)


class PipelineLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that injects pipeline metadata into every log record.

    Useful for tracking a full user request across multiple NLP stages
    (language detection → intent → emotion → RAG).

    Extra context is automatically attached to every log entry.
    """

    def process(self, msg, kwargs):
        kwargs.setdefault("extra", {}).update(self.extra)
        return msg, kwargs


def get_pipeline_logger(logger, **context):
    """
    Create a pipeline-aware logger with attached metadata.

    Args:
        logger (logging.Logger): Base logger instance.
        **context: Arbitrary key-value pairs like:
            session_id, detected_lang, emotion, etc.

    Returns:
        PipelineLoggerAdapter: Logger enriched with context.
    """
    return PipelineLoggerAdapter(logger, context)


def _validate(level: str):
    """
    Validate logging level string.

    Args:
        level (str): Logging level provided by user.

    Returns:
        str: Uppercase validated logging level.

    Raises:
        ValueError: If level is not a valid logging level.
    """
    level = level.upper()
    if level not in LEVELS:
        raise ValueError(f"Invalid level: {level}")
    return level