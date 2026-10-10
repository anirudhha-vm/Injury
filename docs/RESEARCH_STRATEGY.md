# Research strategy — decisions for the strongest defensible papers

Date: 2026-10-10. Written for the research team. These are the decisions I was asked to take on
study design and positioning, with the evidence behind each. Each one is reversible; anything
that needs a real-world act (ethics approval, academy agreements, laboratory access) is listed
at the end as still yours.

## The short version

1. **The planned cohort cannot support the planned headline model.** Realistic injury rates
   give about a dozen growth-related injuries in 300 athletes over one season. That supports a
   model with one or two predictor parameters, not a deep conditioned encoder.
2. **Split the work into three papers ordered by what the data can prove**: (A) validity of
   smartphone kinematics across maturity stages, (B1) the uncertainty-aware growth twin, (B2)
   a pre-registered, deliberately small risk model, with the deep model as exploratory.
3. **The novelty that survives scrutiny** is methodological rigour plus three things nobody has
   published: maturity-timing uncertainty carried into risk, validity of phone-based kinematics
   by maturity stage, and maturity estimation in Indian youth athletes.
4. **The Mirwald coefficients remain unverified.** Two papers reproduce the numbers in the brief
   but disagree with each other on a sign and on a variable. Only the original paper settles it.

## What the literature says (checked 2026-10-10)

| Finding | Source | Consequence |
|---|---|---|
| Of 204 sports-injury prediction models in 30 studies, 98% were at high or unclear risk of bias; none was externally validated; 7% of studies reported calibration; 10% had a sample-size calculation; no machine-learning study shared code or hyperparameters | Bullock et al. 2022, *Sports Med* | A pre-registered, calibrated, openly coded model with a sample-size justification is itself unusual in this field |
| 603 time-loss apophyseal injuries in 16,024 player-seasons of male academy football; peak about 0.4 per 1,000 hours at U12–U16; location moves from foot to knee to hip with age | Bath / LJMU group 2025, *Br J Sports Med* 59:1663 | About 4 athletes in 100 per season. This sets the sample-size problem |
| XGBoost on preseason tests in 734 youth footballers reported 85% precision and recall for "any injury" | Rommers et al. 2020, *Med Sci Sports Exerc* | The benchmark reviewers will cite; half the players were injured, so it is a far easier target than growth-related injury, and it has no calibration or external validation |
| Mirwald equations misestimated maturity offset by −1.25 y (boys) and −3.40 y (girls) in Japanese youth; a population-specific equation was needed | Murayama et al. 2023, *Int J Sport Health Sci* 21:1–8 | Direct support for the claim that maturity equations do not transfer across populations |
| Peak height velocity in Indian children occurs at about 10.5–11 y in girls and 12–13.5 y in boys | Indian Pediatrics 2019 (n = 2,949); South and North Indian longitudinal studies | Earlier than the brief's simulation assumption (12 and 14 y). No study of maturity-offset validity in Indian youth athletes was found |
| OpenCap agrees with marker-based capture to an RMSE of about 2.4–6.4° in jump-landing tasks, best in the sagittal plane; validation work is in adults, with separate markerless studies in pre-school children | OpenCap validation papers 2024–2026 | The 5° limit is reasonable for sagittal angles. Agreement by maturity stage in adolescents is unstudied |
| Digital-twin papers in sport are frameworks and narrative reviews; none conditions movement on growth state or handles maturity uncertainty | Reviews 2024–2026 | "Digital twin" alone is not novel and attracts scepticism; the specific mechanism is |
| SITAR has been used to locate PHV in academy footballers, and Bayesian multinomial models relate growth to injury type; no study combines uncertainty propagation, dynamic prediction, calibration and decision curves | Narrative review 2025 (*Sports Med*); Delphi consensus 2024 (*PLOS One*) | The combination in this project is unpublished |

Search results are summaries; every citation needs checking against the paper before use.

## Decision 1 — Sample size governs model complexity

Computed with `scripts/sample_size.py` (Riley et al. criteria; one outcome per athlete-season;
model assumed to reach 15% of the maximum possible R²).

| Athletes with a growth-related injury per season | Athletes needed for 3 / 5 / 10 parameters | Parameters a cohort of 300 / 1,000 / 2,000 supports |
|---|---|---|
| 4% (time-loss apophyseal, published) | 616 / 1,027 / 2,053 | 1 / 4 / 9 |
| 8% (assumed: wider definition) | 406 / 677 / 1,354 | 2 / 7 / 14 |
| 15% (assumed: all problems, ages 11–16) | 301 / 501 / 1,001 | 2 / 9 / 19 |

The synthetic cohort already showed the symptom: with 55 simulated injuries every model was
near chance and badly over-confident.

**Decisions**
- The **confirmatory** risk model has at most the number of parameters the achieved event count
  supports, fixed before outcome data are seen. Its candidate predictors, in order: growth
  velocity, maturity offset, one landing-load measure.
- The conditioned encoder, the boosted model and the nine-way ablation are **exploratory**
  unless the cohort reaches the event count for their size. They stay in the paper, labelled as
  such, with intervals.
- **Recommend to the team:** plan for at least 1,000 athlete-seasons (more academies, or two
  seasons) and record all physiotherapist-confirmed problems from the weekly questionnaire, not
  only time-loss injuries. Recompute after the first season's incidence is known.

## Decision 2 — Three papers instead of two

| Paper | Claim | Why it is strong | Suggested target (verify current quartile in SJR/JCR) |
|---|---|---|---|
| **A** | Smartphone markerless kinematics agree with marker-based capture within a pre-specified limit, and agreement does not degrade in pre-PHV children | Needs 30–40 participants; answerable with certainty; maturity-stratified validity in adolescents and in an Indian sample is unpublished | *Scand J Med Sci Sports*; alternatives *J Sci Med Sport*, *J Biomech* |
| **B1** | An uncertainty-aware growth twin estimates age at PHV from sparse single-season anthropometry better than maturity-offset equations, and shows when it cannot | Continuous outcome, about seven visits per athlete, so 300 athletes is ample; first maturity-timing data in Indian youth athletes; a methodological result (below) | *IEEE J Biomed Health Inform*; alternative *Comput Biol Med* |
| **B2** | A pre-registered, cause-specific, dynamic risk model with propagated maturity uncertainty: discrimination, calibration, net benefit | Honest scale; directly answers Bullock et al. and Bahr; deep model reported as exploratory | *Br J Sports Med* or *Scand J Med Sci Sports* if the cohort is large; otherwise *J Sci Med Sport* |

B1 is new relative to the brief, which folded the growth twin into Paper B. It is separated
because it is the contribution the data can fully support, and because development produced a
finding worth reporting on its own:

> With about one season of measurements, an athlete's maturity timing is often genuinely
> two-valued: the same growth speed fits either the rising or the falling side of the growth
> spurt, about two years apart. Standard samplers silently pick one. Integrating timing out
> exactly represents both, and shows that uncertainty is only trustworthy within about three
> years of PHV.

That is a concrete, reproducible statement about the limits of single-season maturity
estimation, with a method that handles it.

## Decision 3 — What the papers claim as novel, and what they do not

Claim:
1. Maturity-timing uncertainty quantified per athlete and carried into predicted risk.
2. A demonstration of where single-season maturity estimation fails, and a model that reports
   the failure instead of hiding it.
3. Cohort-specific correction of maturity-offset equations in a population they were not derived
   on (Indian youth).
4. Validity of smartphone kinematics by maturity stage.
5. Reporting discipline the field lacks: pre-registration, sample-size-justified complexity,
   calibration, decision curves, athlete-level uncertainty, open code and configuration.

Do not claim:
- That the model predicts injury well. Say what the data show, with intervals.
- That FiLM conditioning is better than concatenation, unless the cohort is large enough to show
  it. On synthetic data any difference is noise and partly my own construction.
- "Digital twin" as the contribution. Keep the term for the growth model only, where it is
  accurate: a persistent per-athlete state updated with each measurement, with a forecast.
- Mechanistic load proxies as loads. They are geometry-based proxies, not validated.

## Decision 4 — The remaining growth-twin calibration gap

Intervals for age at PHV cover the truth 88–89% of the time within three years of PHV and 53%
for children more than three years before it.
- **Pre-register the restriction**: uncertainty claims apply within three years of PHV; younger
  children are reported separately and flagged as low-information wherever results are shown.
- **Run the one untested fix**: spline flexibility extended to nine years before PHV. Started
  2026-10-10; the result will be added to `docs/plans/M2_done.md` whichever way it comes out.
- Not pursuing further variants without real data: each costs about three hours and the
  synthetic curves are themselves an assumption.

## Decision 5 — Thresholds and limits

- **Risk thresholds are elicited, not chosen by me.** Before data lock, ask each academy
  physiotherapist one structured question: how many unnecessary reviews are acceptable to find
  one athlete who would go on to be injured? The answers define the threshold range for decision
  curves and the thresholds for sensitivity and specificity. A number picked by an analyst has
  no clinical meaning. Until then the code reports no threshold-based measures.
- **Agreement limit for Paper A**: RMSE below 5° for sagittal knee and hip angles at task events
  (primary), consistent with published markerless accuracy. Frontal-plane knee angle is a
  secondary endpoint with the same limit, with lower agreement expected.

## Decision 6 — Sports

Recommend **football and cricket**. Football gives comparability with nearly all the published
growth-and-injury evidence. Cricket is under-studied, is where Indian academies have scale, and
adolescent fast bowlers have a recognised bone-stress problem of the lower back that belongs in
the growth-related outcome group. Depends on which academies agree.

## Decision 7 — Simulation study

- Replace the brief's assumed age at PHV (12 and 14 y) with Indian reference values at the next
  full re-run, and cite them.
- Make simulated training load persist from week to week; at present load in one week says
  nothing about the next, so no exposure effect can be tested.
- Report the simulation at several cohort sizes, to show how performance depends on the number
  of events. That figure makes the sample-size argument visually.

## Verification of the maturity equations

| Source | Magnitudes | Signs | Third term |
|---|---|---|---|
| Brief | — | boys −0.001663; girls −0.002658 | age × sitting height |
| Cossio-Bolaños et al. 2021, *Endocrinol Diabetes Nutr* 68:689–98 | all match | match the brief | age × **standing** height |
| Murayama et al. 2023, *Int J Sport Health Sci* 21:1–8 (page read directly) | all match | both printed as **positive** | age × sitting height |

Moore's girls equation and the stature-only boys variant match the first source exactly.
Every magnitude is corroborated twice. The two sources contradict each other on the signs and
on the variable in the third term, so at least one contains a misprint, and secondary sources
cannot settle it. **The verification gate stays closed.** Checking Table 3 of Mirwald et al.
(2002) and the equations in Moore et al. (2015) takes minutes with the PDFs.

## Still yours
- The original Mirwald and Moore PDFs, and the Khamis–Roche table.
- Ethics approval, consent process, and whether any data may be held in a cloud database.
- Which academies and how many seasons: this decides whether B2 is a confirmatory paper.
- Laboratory and marker-based system for Paper A.
- Two or three adult test clips, so the video step can be run for the first time.
- The physiotherapists' answers that set the risk thresholds.
- Journal choice, after checking current quartiles and scope.

## Sources
- Bullock GS et al. (2022) Just how confident can we be in predicting sports injuries? *Sports Med*. https://keele-repository.worktribe.com/output/423719/
- Distal-to-proximal progression of apophyseal injuries with age in male youth academy footballers (2025) *Br J Sports Med* 59:1663. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12772586/
- Rommers N et al. (2020) A machine learning approach to assess injury risk in elite youth football players. *Med Sci Sports Exerc*. https://pure.amsterdamumc.nl/en/publications/a-machine-learning-approach-to-assess-injury-risk-in-elite-youth-/
- Murayama R, Kigoshi K, Sugiura K (2023) *Int J Sport Health Sci* 21:1–8. https://www.jstage.jst.go.jp/article/ijshs/21/0/21_202206/_pdf
- Cossio-Bolaños M et al. (2021) *Endocrinol Diabetes Nutr* 68:689–98. https://www.elsevier.es/en-revista-endocrinologia-diabetes-nutricion-english-ed--413-articulo-equations-predicting-maturity-status-validation-S2530018021001591
- Height velocity percentiles in Indian children aged 5–17 years (2019) *Indian Pediatr*. https://pubmed.ncbi.nlm.nih.gov/30806356/
- Growth velocity in South Indian children between 3 and 18 years. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10799202/
- Validity and reliability of OpenCap during squatting, hopping, jumping and side-stepping (2024) *J Sports Sci*. https://www.tandfonline.com/doi/full/10.1080/02640414.2024.2415233
- Concurrent validation of OpenCap during a drop jump (2026) *Sci Rep*. https://www.nature.com/articles/s41598-026-44758-0
- Digital twins in sport: concepts, taxonomies, challenges. https://www.sciencedirect.com/science/article/pii/S0957417424019717
- The association of growth and maturation with injury in academy soccer players: a narrative review (2025) *Sports Med*. https://link.springer.com/article/10.1007/s40279-025-02340-0
- Consensus on maturity-related injury risks and prevention in youth soccer: a Delphi study (2024) *PLOS One*. https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0312568
- Riley RD et al. (2020) Calculating the sample size required for developing a clinical prediction model. *BMJ* 368:m441; `pmsampsize` documentation. https://rdrr.io/pkg/pmsampsize/man/pmsampsize.html
