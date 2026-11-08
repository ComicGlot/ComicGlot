"""Command-line argument types."""
import argparse

from comicglot.utils.cli import optional_name


def test_translator_none_can_be_given_explicitly():
    p = argparse.ArgumentParser()
    p.add_argument("--translator", type=optional_name, default=None, choices=[None, "claude"])
    assert p.parse_args(["--translator", "None"]).translator is None
    assert p.parse_args(["--translator", "none"]).translator is None
    assert p.parse_args(["--translator", "claude"]).translator == "claude"
    assert p.parse_args([]).translator is None
