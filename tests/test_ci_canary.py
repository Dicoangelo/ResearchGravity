"""Temporary canary — asserts CI can fail. Deleted immediately after."""


def test_ci_can_actually_fail():
    assert False, "deliberate failure: verifying the CI test gate is real"
