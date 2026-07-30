"""Test configuration for extension tests."""

# Async mode and marker registration now live in pytest.ini.
#
# This module previously carried a pytest_configure that registered an "asyncio"
# marker under the comment "Configure pytest-asyncio to auto mode". That comment
# described an intent the code did not implement — addinivalue_line registers a
# marker name, it does not set asyncio_mode — so the suite ran in strict mode
# while appearing to be configured for auto. Declaring it in pytest.ini removes
# the ambiguity.
