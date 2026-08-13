"""Isolated staged experimental-touch evidence pipeline.

This package deliberately does not import the terpene ranking/model code and never writes to
production databases. Candidate sources are opened read-only; evidence and run state live in
separate SQLite databases selected by a profile.
"""

__version__ = "0.1.0"
PROTOCOL_ID = "experimental-touch-cascade-v1-20260813"
