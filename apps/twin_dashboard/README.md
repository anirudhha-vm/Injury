# YouthTwin research dashboard

Local viewer for one athlete's digital twin: growth curve with uncertainty, 4-week injury risk
per type, the inputs behind the latest estimate, a stick-figure replay of a screening trial, and
decision-support prompts.

```bash
make dashboard-data   # build the data bundle (needs saved growth posteriors: make sim-study-growth)
make dashboard        # opens http://localhost:8501
```

- Runs on your machine only. No login, hosting or monitoring: it is a research viewer.
- Reads `outputs/dashboard/bundle` (override with `YOUTHTWIN_DASHBOARD_BUNDLE`).
- Shows keypoints, never video.
- With the synthetic cohort every page is marked SYNTHETIC, and the risk models are near chance
  there, so the pages show how results would be presented, not that they are useful.
