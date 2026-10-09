"""The paper/ results guard: untagged numbers fail, tagged numbers and placeholders pass."""

from __future__ import annotations

from pathlib import Path

import pytest

from youthtwin.utils.config import repo_root
from youthtwin.utils.results_guard import scan_generated_dir, scan_paper_dir, scan_text

P = Path("paper/x.md")


@pytest.mark.parametrize(
    "line",
    [
        "The model reached an AUC of 0.82.",
        "Coverage was 91% on the cohort.",
        "We recruited n = 300 athletes.",
        "RMSE < 5 degrees was observed.",
        "Expected AUC 0,82",
    ],
)
def test_untagged_numbers_are_flagged(line: str) -> None:
    violations = scan_text(line, P)
    assert len(violations) == 1
    assert violations[0].reason == "untagged number"
    assert "paper/x.md:1" in violations[0].format()


@pytest.mark.parametrize(
    "line",
    [
        "AUC was [[RESULT: time-dependent AUC, growth-related, full model]].",
        "AUC was [[RUN:abc123|0.82]] (95% CI [[RUN:abc123|0.75-0.88]])".replace("95% CI ", ""),
        "Agreement threshold [[SPEC: RMSE < 5 deg]] was pre-specified.",
        "Hall et al. followed [[CITED: n = 501]] academy players.",
        "Follow-up is 9-12 months with screening every 6-8 weeks in athletes aged 8-18 y.",
        "See `threshold = 0.5` in the config and https://example.org/v1.2/page.",
        "<!-- draft note: 0.9 -->",
    ],
)
def test_tagged_or_harmless_text_passes(line: str) -> None:
    assert scan_text(line, P) == []


def test_run_tag_requires_a_run_id() -> None:
    violations = scan_text("AUC was [[RUN:|0.82]].", P)
    assert [v.reason for v in violations] == ["RUN tag without a run ID"]


def test_code_fences_and_references_are_ignored() -> None:
    text = "\n".join(
        [
            "```",
            "auc = 0.82",
            "```",
            "## References",
            "Mirwald RL (2002) Med Sci Sports Exerc 34(4):689-694. doi:10.1097/x p = 0.05",
            "## Results",
            "AUC 0.82",
        ]
    )
    violations = scan_text(text, P)
    assert [v.line for v in violations] == [7]


def test_generated_files_need_provenance_sidecar(tmp_path: Path) -> None:
    figures = tmp_path / "figures"
    figures.mkdir()
    (figures / ".gitkeep").touch()
    (figures / "fig1.png").touch()
    (figures / "fig2.png").touch()
    (figures / "fig2.png.provenance.json").write_text("{}", encoding="utf-8")
    violations = scan_generated_dir(figures)
    assert [v.path.name for v in violations] == ["fig1.png"]
    assert scan_generated_dir(tmp_path / "missing") == []


def test_scan_paper_dir_combines_sources_and_generated(tmp_path: Path) -> None:
    (tmp_path / "tables").mkdir()
    (tmp_path / "tables" / "t1.md").write_text("AUC 0.82", encoding="utf-8")
    (tmp_path / "draft.md").write_text("AUC 0.82\n", encoding="utf-8")
    (tmp_path / "notes.bin").write_bytes(b"0.82")
    reasons = sorted(v.reason for v in scan_paper_dir(tmp_path))
    assert reasons == ["missing provenance sidecar", "untagged number"]
    with pytest.raises(FileNotFoundError):
        scan_paper_dir(tmp_path / "nope")


def test_repo_paper_dir_is_clean() -> None:
    violations = scan_paper_dir(repo_root() / "paper")
    assert violations == [], "\n".join(v.format() for v in violations)
