"""Test configuration for extension tests."""

import pytest


# Configure pytest-asyncio to auto mode
def pytest_configure(config):
    config.addinivalue_line("markers", "asyncio: mark test as async")
