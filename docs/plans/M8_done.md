# M8 done — 2026-10-10: research dashboard and decision-support prompts

## Acceptance criteria
| Criterion | Result |
|---|---|
| Dashboard runs on the synthetic cohort | ✅ `make dashboard-data` builds the bundle for all 300 athletes in under a minute; the app runs headlessly without exceptions (tested, including switching athlete) |
| All text is decision-support framed | ✅ Every prompt the rules can produce contains "consider", cites existing evidence entries, and contains no diagnostic or prescriptive wording (tested by triggering every rule) |
| `make test lint` green | ✅ 430 passed, 2 deselected |

## How to see it
```bash
make dashboard-data   # needs saved growth posteriors from `make sim-study-growth`
make dashboard        # opens http://localhost:8501 in your browser
```
Four tabs per athlete: **Growth** (measurements, fitted curve with band, age at peak height
velocity with interval, velocity), **Injury risk** (4-week risk per injury type over time with
an uncertainty band and recorded injuries marked; the inputs behind the latest estimate),
**Screening replay** (stick figure from keypoints), **Decision support** (prompts with evidence).

## Safeguards built in
- The page states at the top that the data are synthetic and that the models are near chance on
  this cohort; every figure carries the SYNTHETIC watermark.
- Risk shown for an athlete comes from a model fitted without that athlete.
- No growth or maturity value alone can trigger an elevated-risk prompt (tested over a grid).
  v2 had a rule that did exactly that.
- A prompt with unacceptable wording stops the bundle build instead of reaching the screen.
- The bundle contains no names, full dates of birth, or video references (tested).

## Not done / limits
- **The evidence statements are paraphrased from the project brief and are not verified against
  the papers** (`verified_by_human: false`, shown on screen).
- Prompt thresholds (top 10% of the cohort, 7 cm/year, 10 weeks) are placeholders; whether and
  how prompts are shown at all is a team decision.
- The risk model shown is the penalised-logistic hazard; the neural model is not wired in.
- The uncertainty band reflects growth-timing uncertainty only, not model uncertainty.
- Works on synthetic cohorts only: real data would need keypoints and posteriors from the real
  pipeline, and a governance decision about who may view it.
- No authentication, hosting or monitoring, by design.

## Next: M9
Paper outlines and methods text drafted from the configs, with results as placeholders.
