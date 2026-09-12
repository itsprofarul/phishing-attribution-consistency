"""Audit the generated supplementary files for identifying content.

Springer's double-anonymous track means the ESM files travel to reviewers on
their own, so a name in a PDF info dictionary de-anonymises the submission just
as surely as one on the title page. Four layers are checked, because each can
carry a string the others miss:

  1. raw bytes            -- info dictionary, XMP, embedded font names, fragments
  2. inflated streams     -- anything compressed out of layer 1
  3. visible text         -- the glyph runs a reader actually sees, decoded from
                             the [ (g) kern (g) ... ] TJ arrays inside BT/ET
  4. PDF info dictionary  -- read back from the written file, not from the
                             in-memory object we set

Layer 2 is a superset of layer 3; layer 3 exists so a clean result is legible
rather than merely asserted.

Exits non-zero on any finding, so it can gate a release step.
"""
import re
import sys
import zlib
from pathlib import Path

BS = chr(92)  # backslash, spelled out to keep the regexes below readable

TERMS = ["Prof CSCyber", "itsprofarul", "Natarajan", "Samarkand",
         "github.com", "C:" + BS]
EXTRA = [("absolute POSIX path", re.compile(b"/Users/|/home/")),
         ("email address",
          re.compile(b"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+" + BS.encode() + b".[A-Za-z]{2,}"))]
INFO_KEYS = (b"Title", b"Author", b"Subject", b"Keywords", b"Creator",
             b"Producer", b"CreationDate", b"ModDate")

OUT = Path("submission/supplementary")


def inflate_all(raw: bytes) -> bytes:
    """Concatenate every FlateDecode stream we can inflate."""
    chunks = []
    for m in re.finditer(b"stream" + bytes([13]) + b"?" + bytes([10]), raw):
        start = m.end()
        end = raw.find(b"endstream", start)
        if end == -1:
            continue
        try:
            chunks.append(zlib.decompress(raw[start:end]))
        except Exception:
            continue
    return b"".join(chunks)


def _literals(chunk: bytes) -> bytes:
    """Concatenate every PDF string literal, honouring backslash escapes."""
    out, i, n = bytearray(), 0, len(chunk)
    while i < n:
        if chunk[i:i + 1] != b"(":
            i += 1
            continue
        i, depth = i + 1, 1
        while i < n and depth:
            c = chunk[i:i + 1]
            if c == bytes([92]):
                out += chunk[i + 1:i + 2]
                i += 2
                continue
            if c == b"(":
                depth += 1
            elif c == b")":
                depth -= 1
                if not depth:
                    i += 1
                    break
            if depth:
                out += c
            i += 1
    return bytes(out)


def shown_text(raw: bytes) -> bytes:
    """The text a reader sees, reconstructed from the page content streams."""
    runs = []
    for m in re.finditer(b"stream" + bytes([13]) + b"?" + bytes([10]), raw):
        start = m.end()
        end = raw.find(b"endstream", start)
        if end == -1:
            continue
        try:
            blob = zlib.decompress(raw[start:end])
        except Exception:
            continue
        for tb in re.finditer(b"BT(.*?)ET", blob, re.S):
            run = _literals(tb.group(1))
            if run:
                runs.append(run)
    return bytes([10]).join(runs)


def main() -> int:
    findings = []
    files = sorted(p for p in OUT.glob("*") if p.is_file())
    print("auditing {} file(s) in {}".format(len(files), OUT.as_posix()))
    print()

    for f in files:
        raw = f.read_bytes()
        layers = {"raw bytes": raw}
        if f.suffix.lower() == ".pdf":
            layers["inflated streams"] = inflate_all(raw)
            layers["visible text"] = shown_text(raw)

        hits = []
        for term in TERMS:
            pat = re.compile(re.escape(term).encode(), re.I)
            for layer, blob in layers.items():
                if pat.search(blob):
                    hits.append("{!r} in {}".format(term, layer))
        for label, pat in EXTRA:
            for layer, blob in layers.items():
                m = pat.search(blob)
                if m:
                    hits.append("{} {!r} in {}".format(label, m.group(0)[:40], layer))

        seen = "  ({:,} chars visible text)".format(
            len(layers["visible text"])) if "visible text" in layers else ""
        print("  [{}] {}  ({:,.1f} KB){}".format(
            "CLEAN" if not hits else "FOUND", f.name, f.stat().st_size / 1024, seen))
        for h in hits:
            print("           -> " + h)
            findings.append((f.name, h))

    print()
    print("PDF metadata as written:")
    for f in sorted(OUT.glob("*.pdf")):
        raw = f.read_bytes()
        print("  " + f.name)
        found = False
        for key in INFO_KEYS:
            m = re.search(b"/" + key + b"" + rb"\s*\((.*?)\)", raw, re.S)
            if m:
                found = True
                print("    /{:<13} {!r}".format(
                    key.decode(), m.group(1).decode("latin-1", "replace")))
        if not found:
            print("    (no info dictionary entries found)")

    print()
    print("RESULT: " + ("no identifying content found" if not findings
                        else "{} finding(s) -- see above".format(len(findings))))
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
