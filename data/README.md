# data/

Everything here is git-ignored except this README, `schema/`, and `.gitkeep` files.

| Directory | Contents | Rule |
|---|---|---|
| `schema/` | SQL DDL, pydantic models, data dictionary (M1) | tracked |
| `raw_video_LOCAL_ONLY/` | raw screening video | **never committed, never uploaded, never copied to cloud** |
| `keypoints/` | pose keypoints (parquet) | derived; pseudonymous athlete UUIDs only |
| `interim/`, `processed/` | pipeline intermediates and model-ready tables | derived |
| `synthetic/` | simulated cohorts | every file carries `synthetic=True` |

- Athlete IDs are pseudonymous UUIDs. The ID↔identity map is kept **outside** this repository.
- Date of birth is stored at month precision only.
- Research database default is local DuckDB/SQLite. A cloud database (e.g. Supabase) may hold
  synthetic data only until `TODO(human)`: the ethics committee approves anything further.
