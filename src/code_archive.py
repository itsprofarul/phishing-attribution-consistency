"""ESM_4: an anonymised copy of the repository for double-anonymous review.

Builds submission/supplementary/ESM_4_code.zip. The archive is assembled in a
staging directory, anonymised there, and audited on the staged files before it
is zipped, so what ships is exactly what was checked. The audit runs again on
the extracted archive in `verify_extracted`, because a check of the inputs is
not a check of the artifact.

Anonymisation is three separate things, kept separate on purpose:

  1. exclusion   -- whole trees that must not travel (data, .git, the venv),
                    plus src/identity_terms.py, which necessarily names the
                    strings the tooling protects
  2. rewriting   -- README and LICENSE, where identity is the point of the text
                    and it has to be replaced rather than deleted
  3. redaction   -- logs and caches that recorded an absolute path or a
                    username incidentally; the path becomes a placeholder and
                    the surrounding content is kept

Anything the three passes miss is a hard failure, not a warning: build() raises
rather than shipping an archive that names the author.
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PROJECT_ROOT, banner, get_logger  # noqa: E402

log = get_logger("code_archive")

OUT_DIR = PROJECT_ROOT / "submission" / "supplementary"
ARCHIVE = OUT_DIR / "ESM_4_code.zip"
ARCHIVE_PREFIX = "phishing-attribution-consistency"

EXCLUDE_DIRS = {".git", ".claude", "data", "__pycache__", ".pytest_cache", ".venv"}
# Run logs are excluded outright rather than redacted. A log records whatever
# the pipeline happened to print, including absolute paths and usernames, and
# is the one class of file where new identifying content can appear on any run
# without anyone editing anything.
EXCLUDE_SUFFIXES = {".pyc", ".log"}
# Posix-style, relative to the repository root.
EXCLUDE_PATHS = {
    "submission/supplementary",   # would nest the ESM files inside the archive
    "src/identity_terms.py",      # names the very strings being protected
}

PLACEHOLDER_ROOT = "<project-root>"
PLACEHOLDER_USER = "<user>"
CONTACT_LINE = ("The repository URL and archived-release DOI will be provided on "
                "acceptance.")

# Extensions treated as text for redaction. Everything else is copied verbatim
# and must pass the audit unmodified.
TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".csv", ".log", ".cfg", ".toml",
                 ".yml", ".yaml", ".ini", ".rst", ".bat", ".sh", ".gitignore", ""}

BS = chr(92)


# --------------------------------------------------------------------------- #
# Terms
# --------------------------------------------------------------------------- #
def _identity_patterns() -> list[tuple[str, re.Pattern]]:
    """Author terms plus the generic path and email patterns, as bytes regexes."""
    try:
        from src.identity_terms import IDENTITY_TERMS
    except ImportError:
        from identity_terms import IDENTITY_TERMS

    pats = [(what, re.compile(pat.encode(), re.I)) for pat, what in IDENTITY_TERMS]
    pats += [
        ("absolute Windows path", re.compile(rb"[A-Za-z]:" + re.escape(BS).encode()
                                             + rb"[A-Za-z0-9_. -]{2,}")),
        # Alternation, not the literal prefixes: this module is itself archived.
        ("absolute Windows path", re.compile(rb"[A-Za-z]:/(?:Users|home)/")),
        ("absolute POSIX path", re.compile(rb"/(?:Users|home)/")),
        ("email address",
         re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ]
    return pats


def _redactions() -> list[tuple[re.Pattern, str]]:
    """Literal rewrites applied to text files: paths and usernames."""
    try:
        from src.identity_terms import PROJECT_PATH_MARKERS, USERNAME_MARKERS
    except ImportError:
        from identity_terms import PROJECT_PATH_MARKERS, USERNAME_MARKERS

    rules = []
    # Longest first: the full absolute path must go before the bare username.
    for marker in sorted(PROJECT_PATH_MARKERS, key=len, reverse=True):
        drive = r"[A-Za-z]:" + re.escape(BS) + r"?"
        users = r"(?:Users|home)[" + re.escape(BS) + r"/][^" + re.escape(BS) + r"/]+[" \
                + re.escape(BS) + r"/]"
        rules.append((re.compile(drive + users + re.escape(marker), re.I),
                      PLACEHOLDER_ROOT))
        rules.append((re.compile(re.escape(marker), re.I), PLACEHOLDER_ROOT))
    for name in sorted(USERNAME_MARKERS, key=len, reverse=True):
        rules.append((re.compile(re.escape(name), re.I), PLACEHOLDER_USER))
    # Any remaining absolute home path, whoever it belongs to.
    rules.append((re.compile(r"[A-Za-z]:" + re.escape(BS) + r"Users" + re.escape(BS)
                             + r"[^" + re.escape(BS) + r"]+", re.I), PLACEHOLDER_ROOT))
    rules.append((re.compile(r"/(?:home|Users)/[^/ \"']+"), PLACEHOLDER_ROOT))
    # Bare prefix with nothing after it: appears in older run-log lines that
    # quoted a pattern rather than a path.
    rules.append((re.compile(r"/(?:home|Users)/"), PLACEHOLDER_ROOT + "/"))
    return rules


# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
def _included(path: Path) -> bool:
    rel = path.relative_to(PROJECT_ROOT)
    posix = rel.as_posix()
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIXES:
        return False
    return not any(posix == x or posix.startswith(x + "/") for x in EXCLUDE_PATHS)


def _select() -> list[Path]:
    return sorted(p for p in PROJECT_ROOT.rglob("*") if p.is_file() and _included(p))


# --------------------------------------------------------------------------- #
# Rewriting
# --------------------------------------------------------------------------- #
def anonymise_readme(text: str) -> str:
    """Strip identity from the README, keeping everything a reviewer needs.

    The description, install steps, run instructions, layout, methodological
    notes and dependency list are untouched. Only the archived-release line, the
    citation block and the contact section carry identity, and each is replaced
    in place so the document keeps its shape.
    """
    out = text

    # Archived release DOI, wherever it appears as its own line.
    out = re.sub(r"^.*Archived release.*$\n?", "", out, flags=re.M | re.I)

    # Citation: keep the entry so the work is still citable, drop the author.
    out = re.sub(r"@article\{[^,]+,", "@article{anonymous2026attribution,", out)
    out = re.sub(r"^\s*author\s*=\s*\{[^}]*\},?\s*$\n?",
                 "  author  = {Author names withheld for double-anonymous review},\n",
                 out, flags=re.M)

    # Contact section: replace the body, keep the heading. The trailing-space
    # class is deliberately not \s, which would eat the newline after the
    # heading and leave a doubled blank line.
    out = re.sub(r"^(##[ \t]*Contact)[ \t]*$.*\Z",
                 r"\1" + "\n\n" + CONTACT_LINE + "\n",
                 out, flags=re.M | re.S)

    # Close the gap left where the archived-release line was removed.
    out = re.sub(r"\n{3,}", "\n\n", out)

    header = ("<!-- Anonymised copy prepared for double-anonymous review. Author,\n"
              "     affiliation, repository URL and archived-release DOI are withheld\n"
              "     and will be restored on acceptance. -->\n\n")
    return header + out


def anonymise_license(text: str) -> str:
    """Replace the named copyright holder; the licence grant is unchanged."""
    return re.sub(r"^(Copyright \(c\) \d{4}) .*$",
                  r"\1 The Authors (names withheld for double-anonymous review)",
                  text, flags=re.M)


def _redact_text(raw: bytes, rules) -> tuple[bytes, int]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw, 0
    n = 0
    for pat, repl in rules:
        text, k = pat.subn(repl, text)
        n += k
    return text.encode("utf-8"), n


# --------------------------------------------------------------------------- #
# Audit
# --------------------------------------------------------------------------- #
def audit_tree(root: Path, label: str) -> list[tuple[str, str, str]]:
    """Scan every file under `root` as bytes. Returns a list of findings."""
    pats = _identity_patterns()
    findings = []
    files = sorted(p for p in root.rglob("*") if p.is_file())
    for p in files:
        raw = p.read_bytes()
        for what, pat in pats:
            m = pat.search(raw)
            if m:
                findings.append((p.relative_to(root).as_posix(), what,
                                 m.group(0)[:60].decode("latin-1", "replace")))
    log.info("%s: scanned %d file(s), %d finding(s)", label, len(files), len(findings))
    return findings


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build() -> dict:
    banner(log, "ESM_4  ANONYMISED CODE ARCHIVE")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selected = _select()
    rules = _redactions()

    staging = Path(tempfile.mkdtemp(prefix="esm4_"))
    stage_root = staging / ARCHIVE_PREFIX
    redacted: list[tuple[str, int]] = []
    rewritten: list[str] = []

    try:
        for src in selected:
            rel = src.relative_to(PROJECT_ROOT)
            dst = stage_root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)

            if rel.as_posix() == "README.md":
                dst.write_text(anonymise_readme(src.read_text(encoding="utf-8")),
                               encoding="utf-8")
                rewritten.append("README.md")
                continue
            if rel.as_posix() == "LICENSE":
                dst.write_text(anonymise_license(src.read_text(encoding="utf-8")),
                               encoding="utf-8")
                rewritten.append("LICENSE")
                continue

            raw = src.read_bytes()
            if src.suffix.lower() in TEXT_SUFFIXES:
                raw, n = _redact_text(raw, rules)
                if n:
                    redacted.append((rel.as_posix(), n))
            dst.write_bytes(raw)

        findings = audit_tree(stage_root, "staged tree")
        if findings:
            # Printed, not logged. log.error would write the matched literal into
            # results/phase_a.log, which this archive contains -- a failed build
            # would contaminate the very tree the next build has to clean.
            for rel, what, sample in findings[:40]:
                print("  {}: {} {!r}".format(rel, what, sample))
            raise ValueError(
                f"refusing to build ESM_4: {len(findings)} identifying finding(s) "
                "in the staged tree")

        staged = sorted(p for p in stage_root.rglob("*") if p.is_file())
        uncompressed = sum(p.stat().st_size for p in staged)

        tmp_zip = ARCHIVE.with_suffix(".zip.tmp")
        with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            for p in staged:
                # Fixed timestamp: a mtime is metadata a reviewer does not need.
                info = zipfile.ZipInfo(p.relative_to(staging).as_posix(),
                                       date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, p.read_bytes())
        tmp_zip.replace(ARCHIVE)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    result = {"path": ARCHIVE, "files": len(staged), "uncompressed": uncompressed,
              "compressed": ARCHIVE.stat().st_size,
              "redacted": redacted, "rewritten": rewritten,
              "excluded_dirs": sorted(EXCLUDE_DIRS),
              "excluded_paths": sorted(EXCLUDE_PATHS)}

    log.info("%d file(s), %.1f MB uncompressed -> %.1f MB zipped",
             result["files"], uncompressed / 1024 / 1024,
             result["compressed"] / 1024 / 1024)
    if rewritten:
        log.info("rewritten: %s", ", ".join(rewritten))
    for rel, n in redacted:
        log.info("redacted %-42s %d substitution(s)", rel, n)
    return result


def verify_extracted() -> list:
    """Extract the built archive and audit what a reviewer would actually see."""
    banner(log, "AUDIT OF EXTRACTED ARCHIVE", "-")
    tmp = Path(tempfile.mkdtemp(prefix="esm4_verify_"))
    try:
        with zipfile.ZipFile(ARCHIVE) as z:
            z.extractall(tmp)
        findings = audit_tree(tmp, "extracted archive")
        for rel, what, sample in findings:
            print("  {}: {} {!r}".format(rel, what, sample))
        log.info("RESULT: %s", "no identifying content found" if not findings
                 else f"{len(findings)} finding(s)")
        return findings
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    build()
    if verify_extracted():
        sys.exit(1)


if __name__ == "__main__":
    main()
