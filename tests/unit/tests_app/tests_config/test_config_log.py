import logging
import pytest

from app.config.config_log import setup_logging


class TestSetupLogging:
    """Test class for setup_logging function."""

    def test_setup_logging_initial_call(self, caplog):
        """Test that setup_logging configures logging on initial call."""
        # Clear existing handlers to simulate fresh start
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        root_logger.handlers.clear()

        try:
            setup_logging()

            # Check that handlers are added
            assert len(root_logger.handlers) == 2
            # One StreamHandler, one FileHandler
            handler_types = [type(h).__name__ for h in root_logger.handlers]
            assert 'StreamHandler' in handler_types
            assert 'FileHandler' in handler_types

            # Check log level
            assert root_logger.level == logging.INFO

        finally:
            # Restore original handlers
            root_logger.handlers = original_handlers

    def test_setup_logging_subsequent_calls_no_duplicate_handlers(self, caplog):
        """Test that subsequent calls to setup_logging do not add duplicate handlers."""
        # Ensure logging is set up
        setup_logging()
        initial_count = len(logging.getLogger().handlers)

        # Call again
        setup_logging()

        # Should not add more handlers
        assert len(logging.getLogger().handlers) == initial_count