"""Compatibility import namespace for frozen pre-BRIDGE reproduction code.

Current scientific implementation lives in :mod:`projects.active.bridge`.
This package contains no implementation of its own; its module search path is
redirected to BRIDGE so historical imports remain executable without duplicating
or forking current source.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / "bridge")]
