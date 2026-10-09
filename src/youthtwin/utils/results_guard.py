"""Guard against hand-typed results in ``paper/`` (working rule 4, v2 "Expected AUC" defect).

Every statistic-like number in a paper source file must sit inside one of these tags:

- ``[[RESULT: description]]``   placeholder, no number yet
- ``[[RUN:<run_id>|<value>]]``  number written by a script, traceable to a run
- ``[[SPEC: text]]``            pre-specified design value that mirrors ``configs/``
- ``[[CITED: text]]``           value quoted from the literature (with a citation nearby)

Statistic-like means a decimal (``0.82``), a percentage (``85%``) or a comparison/assignment
to a number (``n = 501``, ``< 5``). Plain integers in prose ("4 weeks", "8-18 y"), code spans,
URLs and anything under a "References" heading are ignored.

Generated files in ``paper/figures`` and ``paper/tables`` must have a ``<file>.provenance.json``
sidecar (written by ``scripts/make_figures.py`` from M7).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

TEXT_SUFFIXES = frozenset({".md", ".tex", ".txt"})
GENERATED_DIRS = ("figures", "tables")
PROVENANCE_SUFFIX = ".provenance.json"
IGNORED_GENERATED_NAMES = frozenset({".gitkeep", "README.md"})

_RUN_TAG = re.compile(r"\[\[RUN:(?P<run_id>[^|\]]*)\|[^\]]*\]\]")
_ALLOWED_TAG = re.compile(r"\[\[(?:RESULT|RUN|SPEC|CITED):[^\]]*\]\]")
_NOISE = re.compile(r"`[^`]*`|<!--.*?-->|https?://\S+|doi:\S+", re.IGNORECASE)
_SUSPICIOUS = re.compile(r"\d+[.,]\d+|\d+\s?%|[=<>≤≥≈]\s*-?\d")
_HEADING = re.compile(r"^\s*#+\s*(?P<title>.*)$")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass(frozen=True)
class Violation:
    path: Path
    line: int
    reason: str
    text: str

    def format(self) -> str:
        return f"{self.path}:{self.line}: {self.reason}: {self.text.strip()}"


def scan_text(text: str, path: Path) -> list[Violation]:
    """Return violations for one paper source file."""
    violations: list[Violation] = []
    in_fence = False
    in_references = False
    for number, line in enumerate(text.splitlines(), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        heading = _HEADING.match(line)
        if heading:
            in_references = "reference" in heading.group("title").lower()
        if in_references:
            continue
        for run_tag in _RUN_TAG.finditer(line):
            if not run_tag.group("run_id").strip():
                violations.append(Violation(path, number, "RUN tag without a run ID", line))
        stripped = _NOISE.sub(" ", _ALLOWED_TAG.sub(" ", line))
        if _SUSPICIOUS.search(stripped):
            violations.append(Violation(path, number, "untagged number", line))
    return violations


def scan_generated_dir(directory: Path) -> list[Violation]:
    """Every generated figure/table needs a provenance sidecar."""
    violations: list[Violation] = []
    if not directory.is_dir():
        return violations
    for file in sorted(p for p in directory.rglob("*") if p.is_file()):
        if file.name in IGNORED_GENERATED_NAMES or file.name.endswith(PROVENANCE_SUFFIX):
            continue
        sidecar = file.with_name(file.name + PROVENANCE_SUFFIX)
        if not sidecar.is_file():
            violations.append(Violation(file, 0, "missing provenance sidecar", sidecar.name))
    return violations


def scan_paper_dir(paper_dir: Path) -> list[Violation]:
    """Scan all paper sources and generated outputs under ``paper_dir``."""
    if not paper_dir.is_dir():
        raise FileNotFoundError(f"Paper directory not found: {paper_dir}")
    generated = [paper_dir / name for name in GENERATED_DIRS]
    violations: list[Violation] = []
    for file in sorted(p for p in paper_dir.rglob("*") if p.is_file()):
        if any(parent in file.parents for parent in generated):
            continue
        if file.suffix.lower() in TEXT_SUFFIXES:
            violations.extend(scan_text(file.read_text(encoding="utf-8"), file))
    for directory in generated:
        violations.extend(scan_generated_dir(directory))
    return violations
