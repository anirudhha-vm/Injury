# data/schema/

`schema.sql` is the DDL; `src/youthtwin/io/schema.py` holds the matching validation models;
`docs/data_dictionary.md` documents every field and unit. `tests/test_schema.py` fails if the
DDL and the models drift apart.

Tables (brief §6.10): `athletes`, `anthropometrics`, `screening_sessions`, `trials`,
`task_features`, `exposure_weekly`, `ostrc_weekly`, `injuries`, `growth_posteriors`.
