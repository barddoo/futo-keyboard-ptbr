"""Print one field of the active language config (paths relative to the
project root). Used by the Makefile and the shell scripts:

    python3 -m langs name
    KEYBOARD_LANG=fr python3 -m langs q6k_gguf
"""
import sys
from pathlib import Path

from . import ROOT, load

if len(sys.argv) != 2:
    raise SystemExit("usage: python3 -m langs <field>")

value = getattr(load(), sys.argv[1])
if isinstance(value, Path):
    value = value.relative_to(ROOT).as_posix()
print(value)
