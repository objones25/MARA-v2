import json
import logging

import pytest

from mara.logging import configure_logging


@pytest.fixture(autouse=True)
def reset_mara_logger():
    """Restore mara logger to pristine state after each test."""
    logger = logging.getLogger("mara")
    original_handlers = logger.handlers[:]
    original_level = logger.level
    original_propagate = logger.propagate
    yield
    logger.handlers = original_handlers
    logger.level = original_level
    logger.propagate = original_propagate


class TestConfigureLogging:
    def test_returns_mara_logger(self):
        log = configure_logging()
        assert log.name == "mara"

    def test_default_level_is_info(self):
        log = configure_logging()
        assert log.level == logging.INFO

    def test_custom_level_debug(self):
        log = configure_logging("DEBUG")
        assert log.level == logging.DEBUG

    def test_custom_level_warning(self):
        log = configure_logging("WARNING")
        assert log.level == logging.WARNING

    def test_invalid_level_raises(self):
        with pytest.raises(ValueError, match="Invalid log level"):
            configure_logging("BOGUS")

    def test_propagate_is_false(self):
        log = configure_logging()
        assert log.propagate is False

    def test_handler_added(self):
        log = configure_logging()
        assert len(log.handlers) >= 1

    def test_idempotent_no_duplicate_handlers(self):
        configure_logging()
        configure_logging()
        log = logging.getLogger("mara")
        assert len(log.handlers) == 1

    def test_handler_outputs_valid_json(self, capsys):
        log = configure_logging()
        log.info("test message")
        captured = capsys.readouterr()
        record = json.loads(captured.err)
        assert record["message"] == "test message"
        assert record["level"] == "INFO"
        assert "timestamp" in record

    def test_json_record_has_logger_name(self, capsys):
        log = configure_logging()
        log.info("hello")
        captured = capsys.readouterr()
        record = json.loads(captured.err)
        assert record["logger"] == "mara"

    def test_does_not_affect_root_logger(self):
        root_handlers_before = logging.root.handlers[:]
        configure_logging()
        assert logging.root.handlers == root_handlers_before
