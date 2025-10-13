# Copyright © 2025 PI & Other Tales Inc.. All Rights Reserved.
from agent.configuration import Configuration


def test_configuration_empty() -> None:
    Configuration.from_runnable_config({})
