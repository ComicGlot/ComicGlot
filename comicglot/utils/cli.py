"""Small argument types shared by the command-line scripts."""
from typing import Optional


def optional_name(value: str) -> Optional[str]:
    """'none' (any case) or an empty string means no value; anything else is kept as given.

    argparse checks choices after the type, so `choices=[None, "claude"]` with this type accepts
    `--translator None` as well as `--translator claude`.
    """
    return None if value.strip().lower() in {"", "none"} else value
