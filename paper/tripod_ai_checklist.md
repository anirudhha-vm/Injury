# TRIPOD+AI checklist — working copy

Collins GS et al. (2024) TRIPOD+AI statement, BMJ. Kept up to date as code is written (N7).

> TODO(human): the item wording below is a paraphrased working summary. Before submission,
> replace it with the official checklist from the TRIPOD+AI publication and confirm item numbers.

Status: `todo` · `partial` · `done`. "Where" points to the code/config/doc that satisfies the item.

| Item | Section | Topic (paraphrased) | Status | Where |
|---|---|---|---|---|
| 1 | Title | Identify as development/evaluation of a prediction model, population, outcome | todo | `paper/paperB_outline.md` (M9) |
| 2 | Abstract | Structured abstract per TRIPOD+AI for Abstracts | todo | M9 |
| 3 | Introduction | Background, rationale, existing models, health inequalities | todo | M9 |
| 4 | Introduction | Objectives | partial | `paper/preregistration.md` |
| 5 | Methods | Source of data, dates, setting | todo | `docs/protocol/` |
| 6 | Methods | Participants: eligibility, setting, treatments | partial | `docs/protocol/` |
| 7 | Methods | Data preparation and quality checking | todo | `src/youthtwin/io/qc.py` (M1), pose QC (M3) |
| 8 | Methods | Outcome definition, assessment, blinding | partial | `configs/models/hazard.yaml`, `docs/protocol/04_injury_surveillance_sop.md` |
| 9 | Methods | Predictors: definition, timing, measurement | partial | `configs/features/`, `configs/growth/` |
| 10 | Methods | Sample size justification | todo | `scripts/sample_size.py` |
| 11 | Methods | Missing data handling | todo | M5 |
| 12 | Methods | Analytical methods: model type, tuning, validation, performance measures | partial | `configs/models/`, `configs/eval/` |
| 13 | Methods | Class imbalance handling | partial | `configs/models/hazard.yaml` |
| 14 | Methods | Fairness approaches | partial | `configs/eval/metrics.yaml` (subgroups) |
| 15 | Methods | Model output and thresholds | todo | M5–M7 |
| 16 | Methods | Differences between development and evaluation data | todo | M7 |
| 17 | Methods | Ethical approval and consent | todo | `docs/ethics/` (DRAFT) |
| 18 | Open science | Funding, conflicts, protocol, registration, data and code sharing | partial | `paper/preregistration.md`, this repository |
| 19 | PPI | Patient and public involvement | todo | TODO(human) |
| 20 | Results | Participant flow and characteristics | todo | M7 |
| 21 | Results | Model development: participants and events per analysis | todo | M7 |
| 22 | Results | Full model specification | todo | M7 |
| 23 | Results | Model performance with CIs, including subgroups | todo | `src/youthtwin/eval/` (M7) |
| 24 | Results | Model updating | todo | M7 |
| 25 | Discussion | Interpretation, including fairness | todo | M9 |
| 26 | Discussion | Limitations | todo | M9 |
| 27 | Discussion | Usability: input data quality, user interaction, future research | todo | M9 |
