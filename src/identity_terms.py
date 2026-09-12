"""The author-identifying strings the anonymity tooling searches for.

Kept in its own module for one reason: the tooling that scrubs and audits has
to name the very strings it is protecting, so `supplementary.py` and
`anon_audit.py` would otherwise be the two files in the repository guaranteed
to leak the author's identity into an anonymised code archive.

This module is excluded from ESM_4. The importers fall back to an empty term
list when it is missing, which is the correct behaviour for a reviewer: there
is no author identity in their copy to protect, so the scrub becomes a no-op
and the audit reports that no terms are configured.

Add a term here and both tools pick it up.
"""
from __future__ import annotations

# Literal strings. Matched case-insensitively by both importers.
IDENTITY_TERMS: list[tuple[str, str]] = [
    (r"Prof\s*[-_]?\s*CSCyber", "author account name"),
    (r"itsprofarul", "author handle"),
    (r"Natarajan", "author surname"),
    (r"Arul\s+Kumar", "author given name"),
    (r"Samarkand", "affiliation"),
    (r"github\.com", "repository URL"),
    (r"zenodo", "archived release DOI"),
]

# Where the author's working copy lives, so logs and caches that recorded an
# absolute path can be rewritten to a placeholder rather than dropped.
PROJECT_PATH_MARKERS: list[str] = [
    "Downloads/[74] - SCI Springer Journal/Code",
    "Downloads" + chr(92) + "[74] - SCI Springer Journal" + chr(92) + "Code",
]
USERNAME_MARKERS: list[str] = ["Prof CSCyber", "PROFCS~1", "Prof-CSCyber"]
